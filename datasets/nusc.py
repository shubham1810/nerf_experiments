import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from .ray_utils import *


class NusDataset(Dataset):
    def __init__(self, root_dir, split="train", img_wh=(400, 225), downscale=1, perturbation=[]):
        self.root_dir = root_dir
        self.split = split
        self.downscale = downscale
        
        self.img_wh = img_wh
        self.define_transforms()
        self.perturbation = perturbation
        
        w, h = self.img_wh
        self.res_wh = (w // self.downscale, h // self.downscale)
        self.res_mat = np.eye(3)
        self.res_mat[0, 0] = self.res_mat[1, 1] = 1/self.downscale
        
        self.read_meta()
        self.white_back = True
    
    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)
        
        self.near = 1.01
        self.far = 100.0
        self.bounds = np.array([self.near, self.far])
        w, h = self.res_wh
        
        if self.split == 'train':
            self.all_rays = []
            self.all_rgbs = []
            for t, frame in enumerate(self.meta['frames']):
                # if t >= 40:
                #     break
                K = self.res_mat @ np.array(frame['intrinsic']).reshape((3, 3))
                focal = K[0, 0]
                pose = np.array(frame['extrinsic']).reshape((4, 4))[:3, :4]
                c2w = torch.FloatTensor(pose)
                
                image_path = os.path.join(self.root_dir, "{}.jpg".format(frame['file_name']))
                img = Image.open(image_path)
                img = img.resize(self.res_wh, Image.LANCZOS)
                img = self.transform(img).view(3, -1).permute(1, 0)
                self.all_rgbs += [img]
                
                # Now for the rays
                directions = get_ray_directions(h, w, K[0, 0])
                rays_o, rays_d = get_rays(directions, c2w)
                # rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                #                                 focal, 1.0, rays_o, rays_d)
                
                # Add to list of all rays
                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),],
                                             1)]
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
    
    def define_transforms(self):
        self.transform = T.ToTensor()
    
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8
        return len(self.meta['frames'])
    
    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx, :8],
                      'rgbs': self.all_rgbs[idx]}
        else:
            # create data for each image separately
            frame = self.meta['frames'][idx]
            
            K = self.res_mat @ np.array(frame['intrinsic']).reshape((3, 3))
            focal = K[0, 0]
            pose = np.array(frame['extrinsic']).reshape((4, 4))[:3, :4]
            c2w = torch.FloatTensor(pose)
            
            image_path = os.path.join(self.root_dir, "{}.jpg".format(frame['file_name']))
            img = Image.open(image_path)
            img = img.resize(self.res_wh, Image.LANCZOS)
            img = self.transform(img).view(3, -1).permute(1, 0)
            valid_mask = (img[0]>0).flatten()
            
            w, h = self.res_wh
            
            # Now for the rays
            directions = get_ray_directions(h, w, K[0, 0])
            rays_o, rays_d = get_rays(directions, c2w)
            # rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
            #                                     focal, 1.0, rays_o, rays_d)
            
            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                             1)
            
            sample = {
                'rays': rays,
                'rgbs': img,
                'c2w': c2w,
                'valid_mask': valid_mask
            }
            
            if self.split == 'train_test':
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask
        return sample
