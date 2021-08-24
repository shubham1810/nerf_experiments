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

        self.loss = loss_dict['llal'](coef=1)

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRFLoss()
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            # self.nerf_fine = NeRF()
            self.nerf_fine = NeRFLoss()
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
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_loss_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
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
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        results = self(rays)

        color_loss, llal_loss = self.loss(results, rgbs)

        if self.current_epoch > 2:
            loss = color_loss + llal_loss
        else:
            loss = color_loss

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        
        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/total_loss', loss)
        self.log('train/pred_loss', results[f'rgb_loss_{typ}'].detach().mean())
        self.log('train/rgb_loss', color_loss)
        self.log('train/learned_loss', llal_loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)

        color_loss, llal_loss = self.loss(results, rgbs)
        loss = color_loss + llal_loss

        log = {'val_total_loss': loss, 'val_rgb_loss': color_loss, 'val_learned_loss': llal_loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        log['val_pred_loss'] = results[f'rgb_loss_{typ}'].mean()
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            diff_img = visualize_depth(((img_gt - img)**2).mean(0))
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)

            # Visualize the loss maps as well
            rgb_loss = visualize_depth(results[f'rgb_loss_{typ}'].view(H, W)) # (3, H, W)

            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            stack_loss = torch.stack([diff_img, rgb_loss, img])
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
            self.logger.experiment.add_images('val/GT_rgbloss_pred',
                                               stack_loss, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_total_loss'] for x in outputs]).mean()
        mean_rgb_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        mean_pred_loss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
        mean_learned_loss = torch.stack([x['val_learned_loss'] for x in outputs]).mean()

        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/total_loss', mean_loss)
        self.log('val/pred_loss', mean_pred_loss)
        self.log('val/rgb_loss', mean_rgb_loss)
        self.log('val/learned_loss', mean_learned_loss)
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
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      plugins=DDPPlugin(find_unused_parameters=True),
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)
    trainer.save_checkpoint(f'ckpts/{hparams.exp_name}/last_epoch.ckpt')


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
