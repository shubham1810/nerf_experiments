import torch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

__all__ = ['render_rays', 'render_2d_rays', 'render_loss_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def sample_pdf2d(bins, weights, N_importance, det=False, eps=1e-5):
    h_, w_, ns = weights.shape
    weights = weights + eps # avoid division by zero error
    pdf = weights / reduce(weights, 'h w n -> h w 1', 'sum')
    cdf = torch.cumsum(pdf, -1)

    if det:
        # Stupid error here if we don't substract with eps. 1 can be included in the
        # matrix and it can give error later.
        u = torch.linspace(eps, 1, N_importance, device=bins.device) - eps
        u = u.expand(h_, w_, N_importance)
        u = u.contiguous()
    else:
        u = torch.rand(h_, w_, N_importance, device=bins.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, ns-1)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'h w n2 c -> h w (n2 c)', c=2)
    gathered = torch.gather(cdf, 2, inds_sampled)
    cdf_g = rearrange(gathered, 'h w (n2 c) -> h w n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 2, inds_sampled), 'h w (n2 c) -> h w n2 c', c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    if (denom<eps).sum() > 0:
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                            # in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)
    
    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        # [CHANGES]: Added a softmax part on the weights here
        wts_sf = torch.nn.functional.softmax(weights*2, dim=-1)
        # print(sigmas.mean(-1).shape, sigmas_sf.shape)
        wts = wts_sf

        # print(reduce(wts, 'n1 n2 -> n1', 'sum'), weights_sum)

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        # rgb_map = reduce(rearrange(wts, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')

        depth_map = reduce(wts*z_vals, 'n1 n2 -> n1', 'sum')
        # depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)

    return results


def render_2d_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    
    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        B = xyz.shape[0]
        out_chunks = []
        
        if typ == 'coarse' and test_time and 'fine' in models:
            xyz_parts = []
            for i in range(B):
                part = xyz[i:i+1]
                _b, _c, _h, _w = part.shape

                part_emb = embedding_xyz(rearrange(part, 'b c h w -> (b h w) c'))
                part_emb = rearrange(part_emb, '(b h w) c -> b c h w', b=_b, h=_h, w=_w)
                xyz_parts += [part_emb]

            out_chunks += [model(torch.cat(xyz_parts, 0), sigma_only=True)]
            
            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, 'b 1 h w -> h w b')
        else:
            dir_embedded = repeat(dir_emb, 'b c h w -> (b B) c h w', B=B)
            xyz_parts = []
            for i in range(B):
                part = xyz[i:i+1]
                _b, _c, _h, _w = part.shape

                part_emb = embedding_xyz(rearrange(part, 'b c h w -> (b h w) c'))
                part_emb = rearrange(part_emb, '(b h w) c -> b c h w', b=_b, h=_h, w=_w)
                xyz_parts += [part_emb]

            input_emb = torch.cat([torch.cat(xyz_parts, 0), dir_embedded], 1)
            out_chunks += [model(input_emb, sigma_only=False)]
            
            out = torch.cat(out_chunks, 0)
            out = rearrange(out, 'b c h w -> h w b c')
            rgbs = out[..., :3]
            sigmas = out[..., 3]
        
        # Now time for processing the inferred data for rendering
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
        deltas = torch.cat([deltas, delta_inf], -1)
        
        # compute alpha
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas * torch.relu(sigmas + noise))
        
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[..., :1]), 1-alphas+1e-10], -1)
        weights = \
            alphas * torch.cumprod(alphas_shifted[..., :-1], -1)
        weights_sum = reduce(weights, 'h w b -> h w', 'sum')
        
        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return
        
        # render colors now
        rgb_map = reduce(rearrange(weights, 'h w b -> h w b 1')*rgbs, 'h w b c -> h w c', 'sum')
        depth_map = reduce(weights * z_vals, 'h w b -> h w', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(-1)
        
        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map
        
        return


    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']
    
    # Decompose the inputs
    h, w, _ = rays.shape
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]
    near, far = rays[..., 6:7], rays[..., 7:8]

    # Apply direction embedding
    _h, _w, _c = rays_d.shape
    dir_emb = embedding_dir(rearrange(rays_d, 'h w c -> (h w) c'))
    dir_emb = rearrange(dir_emb, '(h w) c -> 1 c h w', h=_h, w=_w)

    rays_o = rearrange(rays_o, 'n1 n2 c -> n1 n2 1 c')
    rays_d = rearrange(rays_d, 'n1 n2 c -> n1 n2 1 c')

    # sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[..., :-1] + z_vals[... ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[... ,-1:]], -1)
        lower = torch.cat([z_vals[... ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 c -> n1 n2 c 1')
    xyz_coarse = rearrange(xyz_coarse, 'h w b c -> b c h w')
    # print(xyz_coarse.shape, z_vals.shape, dir_emb.shape)
    
    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time)
    
    if N_importance > 0: # Sample points for the fine model
        z_vals_mid = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        # return z_vals_mid, results['weights_coarse'][..., 1:-1].detach(), N_importance
        z_vals_ = sample_pdf2d(z_vals_mid, results['weights_coarse'][..., 1:-1].detach(),
                            N_importance, det=(perturb==0))
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        
        # prepare fine points
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 c -> n1 n2 c 1')
        xyz_fine = rearrange(xyz_fine, 'h w b c -> b c h w')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)
    
    return results


def render_loss_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays and Learn a Loss term as well by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_val = model(xyz_embedded, sigma_only=True)
                out_chunks += [out_val]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                out_val = model(xyzdir_embedded, sigma_only=False)
                out_chunks += [out_val]

            out = torch.cat(out_chunks, 0)

            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=5)

            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            betas = F.softplus(out[..., 3]) + 0.001
            sigmas = out[..., 4] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]

        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        # print(reduce(wts, 'n1 n2 -> n1', 'sum'), weights_sum)

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals

        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        betas_map = reduce(weights.detach()*betas, 'n1 n2 -> n1', 'sum')

        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        results[f'rgb_beta_{typ}'] = betas_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)

    return results