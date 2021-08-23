import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Graph stuff
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from einops import rearrange, reduce, repeat


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='mean') #  "Avg" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                            ReLU(True),
                            Linear(out_channels, out_channels),
                            ReLU(True))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=3):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, edge_index, batch=None):
        # print("Finding neighbours for: ", x.shape)
        # edge_index = knn_graph(x, self.k, batch, loop=False)
        # print("Found neighbours!")
        return super(DynamicEdgeConv, self).forward(x, edge_index)


def compute_edges(shape):
    """
    [NOTE]: Make sure that the shape is in the form (H x W x B).
    The batch should be last
    """
    # Placeholders for to and from
    from_ = []
    to_ = []

    # create index matrix
    im = np.arange(np.prod(shape)).reshape(shape)
    from_ += [im[:-1].flatten()]
    from_ += [im[1:].flatten()]
    to_ += [im[1:].flatten()]
    to_ += [im[:-1].flatten()]

    from_ += [im[:, :-1].flatten()]
    from_ += [im[:, 1:].flatten()]
    to_ += [im[:, 1:].flatten()]
    to_ += [im[:, :-1].flatten()]
    return np.array([np.concatenate(from_, 0), np.concatenate(to_, 0)])


class NeRFGraph(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFGraph, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding1 = EdgeConv(W + in_channels_dir, W//2)
        self.dir_encoding2 = EdgeConv(W//2, W//2)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb_graph = EdgeConv(W//2, 3)
        self.rgb = nn.Sigmoid()

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')

        # Compute the edges
        edge_index = torch.LongTensor(compute_edges(shape=(h, w, b))).to(x.device)
        
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz

        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
                # print(xyz_.shape)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        sigma = rearrange(sigma, '(b h w) c -> b c h w', b=b, h=h, w=w)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding1(dir_encoding_input, edge_index)
        dir_encoding = self.dir_encoding2(dir_encoding, edge_index)
        rgb = self.rgb(self.rgb_graph(dir_encoding, edge_index))

        rgb = rearrange(rgb, '(b h w) c -> b c h w', b=b, h=h, w=w)
        
        out = torch.cat([rgb, sigma], 1)

        return out


class NeRF2D(nn.Module):
    def __init__(self,
                K=[3, 3, 1, 1, 3, 3, 1, 1], W=256,
                in_channels_xyz=63, in_channels_dir=27,
                skips=[], bn=[],
                ):
        super(NeRF2D, self).__init__()

        self.K = K
        self.D = len(self.K)
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.bn = bn

        # Compute total padding beforehand
        pad = 0
        for i in range(self.D):
            if self.K[i] > 1:
                pad += self.K[i]//2
        print("Padding: {}".format(pad))
        # xyz encoding layers
        for i in range(self.D):
            kern = self.K[i]
            if i == 0:
                layer = nn.Conv2d(in_channels_xyz, W, kern, padding=pad, padding_mode='reflect')
            elif i in skips:
                layer = nn.Conv2d(in_channels_xyz+W, W, kern)
            else:
                layer = nn.Conv2d(W, W, kern)
            
            if i in bn:
                layer = nn.Sequential(layer, nn.ReLU(True), nn.BatchNorm2d(W))
            else:
                layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        
        self.xyz_encoding_final = nn.Conv2d(W, W, 1)

        # Direction encoding layers
        self.dir_encoding = nn.Sequential(
                                    nn.Conv2d(W+in_channels_dir, W, 3, padding=2, padding_mode='reflect'),
                                    nn.ReLU(True),
                                    nn.Conv2d(W, W//2, 3),
                                    nn.ReLU(True),
                                )
        
        # output layers
        self.sigma = nn.Conv2d(W, 1, 1)
        self.rgb = nn.Sequential(
                            nn.Conv2d(W//2, 3, 1),
                            nn.Sigmoid()
                        )
        
    
    def forward(self, x, sigma_only=False):
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=1)
        else:
            input_xyz = x
        
        xyz_ =  input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ =  getattr(self, f"xyz_encoding_{i+1}")(xyz_)
            # print(xyz_.shape)
        
        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma
        
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        # print(xyz_encoding_final.shape, input_dir.shape)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], 1)
        return out


class LossNet(nn.Module):
    def __init__(self, n_features=8, feature_dims=256, rgb_features=[256, 128], h_dim=128):
        super().__init__()

        self.n_features = n_features
        self.feature_dims = feature_dims
        self.h_dim = h_dim
        self.rgb_features = rgb_features

        # loss encoding layers
        for i in range(self.n_features):
            layer = nn.Sequential(
                nn.Linear(self.feature_dims, self.h_dim),
                nn.ReLU(True),
                nn.Dropout(0.21),
            )
            setattr(self, f"loss_encoding_{i+1}", layer)

        # RGB loss encoding layers
        for i in range(len(self.rgb_features)):
            layer = nn.Sequential(
                nn.Linear(self.rgb_features[i], self.h_dim),
                nn.ReLU(True)
            )
            setattr(self, f"rgb_loss_encoding_{i+1}", layer)
        
        # Final loss prediction layers
        self.rgb_loss = nn.Linear(self.h_dim * (self.n_features + len(self.rgb_features)), 1)
    
    def forward(self, features, sigma_only=False):
        if not sigma_only:
            features, color_features = features[:self.n_features], features[self.n_features:]
        else:
            features = features[:self.n_features]

        
        loss_outs = []
        for ix in range(self.n_features):
            z_loss = getattr(self, f"loss_encoding_{ix+1}")(features[ix])
            # print(ix, features[ix].shape, z_loss.shape)
            loss_outs.append(z_loss)
        
        for i in range(len(self.rgb_features)):
            c_loss = getattr(self, f"rgb_loss_encoding_{i+1}")(color_features[i])
            # print(f"rgb_loss_encoding_{i+1}", color_features[i].shape, c_loss.shape)
            loss_outs.append(c_loss)
        
        final_loss_features = torch.cat(loss_outs, 1)
        # print(final_loss_features.shape)
        rgb_loss_pred = F.softplus(self.rgb_loss(final_loss_features))

        return rgb_loss_pred


class NeRFLoss(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFLoss, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3+1), # 3+1 for RGB and uncertainty
                        nn.Sigmoid())
        
        # self.pred_loss = LossNet(n_features=D, feature_dims=W, rgb_features=[W, W//2], h_dim=128)

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        features = []
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)


        rgb_beta = self.rgb(dir_encoding)

        out = torch.cat([rgb_beta, sigma], -1)

        return out
