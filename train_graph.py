import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.plugins import DDPPlugin


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        # print("Hyperparameters:", self.hparams)

        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRFGraph(D=4, skips=[])
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            # self.nerf_fine = NeRF()
            self.nerf_fine = NeRFGraph()
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for bx in range(B):
            results_chunks = render_2d_rays(self.models,
                                self.embeddings,
                                rays[bx],
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                self.hparams.perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back)
            for k, v in results_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        
        return results
    
    def val_forward(self, rays):
        """
        Perform inference on validation data. break inpo parts and
        then reassemble.
        """
        # print(rays.shape)
        results = defaultdict(list)
        if self.val_dataset.apply_skip:
            # Create a template for the outputs
            template_rgb_coarse = torch.zeros_like(rays[..., :3])
            template_rgb_fine = torch.zeros_like(rays[..., :3])
            template_depth_coarse = torch.zeros_like(rays[..., 0])
            template_depth_fine = torch.zeros_like(rays[..., 0])

            # Divide the data samples
            for sx in range(self.val_dataset.skips[0]):
                for sy in range(self.val_dataset.skips[1]):
                    rays_segment = rays[sx::self.val_dataset.skips[0], sy::self.val_dataset.skips[1]]
                    # [TODO]: A very weird "CUDA Assert triggered" error occurs in sample_pdf2d if
                    # I remove this no_grad part. Figure out why!
                    # [UPDATE]: nvm. the error is coming when perturmation!=0. i.e. when doing random sampling.
                    with torch.no_grad():
                        op = render_2d_rays(self.models, #LOL
                                self.embeddings,
                                rays_segment,
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                self.hparams.perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back)
                    template_rgb_coarse[sx::self.val_dataset.skips[0], sy::self.val_dataset.skips[1]] = op['rgb_coarse']
                    template_depth_coarse[sx::self.val_dataset.skips[0], sy::self.val_dataset.skips[1]] = op['depth_coarse']

                    if 'rgb_fine' in op:
                        template_rgb_fine[sx::self.val_dataset.skips[0], sy::self.val_dataset.skips[1]] = op['rgb_fine']
                        template_depth_fine[sx::self.val_dataset.skips[0], sy::self.val_dataset.skips[1]] = op['depth_fine']

            # Now fill all these in results
            results['rgb_coarse'] = template_rgb_coarse
            results['depth_coarse'] = template_depth_coarse
            if 'rgb_fine' in op:
                results['rgb_fine'] = template_rgb_fine
                results['depth_fine'] = template_depth_fine
        elif self.val_dataset.apply_break:
            # Create a template for the outputs
            template_rgb_coarse = torch.zeros_like(rays[..., :3])
            template_rgb_fine = torch.zeros_like(rays[..., :3])
            template_depth_coarse = torch.zeros_like(rays[..., 0])
            template_depth_fine = torch.zeros_like(rays[..., 0])

            # Divide the data into tiles
            for sx in range(self.val_dataset.windows[0]):
                row_start, row_end = sx*self.val_dataset.img_chunk[1], (sx+1)*self.val_dataset.img_chunk[1]
                if sx == self.val_dataset.windows[0]-1:
                    row_end = self.val_dataset.img_wh[1]
                for sy in range(self.val_dataset.windows[1]):
                    col_start, col_end = sy*self.val_dataset.img_chunk[0], (sy+1)*self.val_dataset.img_chunk[0]
                    if sy == self.val_dataset.windows[1]-1:
                        col_end = self.val_dataset.img_wh[0]
                    
                    rays_segment = rays[row_start:row_end, col_start:col_end]
                    with torch.no_grad():
                        op = render_2d_rays(self.models,
                                self.embeddings,
                                rays_segment,
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                self.hparams.perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back)
                    template_rgb_coarse[row_start:row_end, col_start:col_end] = op['rgb_coarse']
                    template_depth_coarse[row_start:row_end, col_start:col_end] = op['depth_coarse']

                    if 'rgb_fine' in op:
                        template_rgb_fine[row_start:row_end, col_start:col_end] = op['rgb_fine']
                        template_depth_fine[row_start:row_end, col_start:col_end] = op['depth_fine']

            # Now fill all these in results
            results['rgb_coarse'] = template_rgb_coarse
            results['depth_coarse'] = template_depth_coarse
            if 'rgb_fine' in op:
                results['rgb_fine'] = template_rgb_fine
                results['depth_fine'] = template_depth_fine



        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        results = self(rays)

        if rgbs.shape[0] == 1:
            rgbs = rgbs.squeeze(0)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self.val_forward(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                        filename='{epoch:d}',
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=1)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=True,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      plugins=DDPPlugin(find_unused_parameters=False),
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)
    trainer.save_checkpoint(f'ckpts/{hparams.exp_name}/last_epoch.ckpt')


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
