import os
import glob
import tqdm
import random
import logging
import gc 

import numpy as np
import imageio, imageio_ffmpeg 
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchmetrics.functional import pearson_corrcoef


import ipdb
from nerf.utils import save_tensor2image, nonzero_normalize_depth, Trainer
from einops import rearrange

import copy
from nerf.utils import custom_meshgrid, safe_normalize

logger = logging.getLogger(__name__)



class DTrainer(Trainer):
    def __init__(self, argv, name, opt, model, guidance, criterion=None, optimizer=None, ema_decay=None, lr_scheduler=None, metrics=[], local_rank=0, world_size=1, device=None, mute=False, fp16=False, max_keep_ckpt=1, workspace='workspace', best_mode='min', use_loss_as_metric=True, report_metric_at_train=False, use_checkpoint="latest", use_tensorboard=True, scheduler_update_every_step=False, **kwargs):
        super().__init__(argv, name, opt, model, guidance, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboard, scheduler_update_every_step, **kwargs)
        self.rgbd_scale = opt.get("rgbd_scale", 1.0)
        
        self.fix_dynamic = opt.fix_dynamic
        if self.fix_dynamic:

            assert opt.backbone == 'grid4d'
            from dnerf.network_4dgrid import NeRFNetwork

            self.dynamic_model = NeRFNetwork(opt)
            # ipdb.set_trace()
            model_state_dict = self.model.state_dict()
            self.dynamic_model.load_state_dict(model_state_dict)
            for p in self.dynamic_model.parameters():
                p.requires_grad = False
            self.dynamic_model.train()
            self.dynamic_model.to(opt.device)

    @torch.no_grad()
    def eval_static_step(self, data, shading):

        rays_o = data['rays_o'] # [B, N, 3] / B,F,N,3
        rays_d = data['rays_d'] # [B, N, 3] / B,F,N,3
        mvp = data['mvp'] # B,4,4 / B,F,4,4
        
        if rays_o.ndim == 4:
            rays_o = rays_o[:, 0]
            rays_d = rays_d[:, 0]
            mvp = mvp[:, 0]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
 
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        # ipdb.set_trace()
        outputs = self.static_model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W, 1)
        if self.opt.normalize_depth: 
            pred_depth = nonzero_normalize_depth(pred_depth)
        if 'normal_image' in outputs: # eval mode no normal image
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        else:
            pred_normal = None 
        
        pred_mask = outputs['weights_sum'].reshape(B, H, W, 1)

        out_dict = {
            'rgb': pred_rgb,
            'depth': pred_depth,
            'normal_image': pred_normal,
            'mask': pred_mask,
        }
        return out_dict

    def train_step(self, data):
        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.images is not None and \
            ((self.global_step < self.opt.known_iters) or (self.global_step % self.opt.known_view_interval == 0))
        # ipdb.set_trace()
        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, 0.2 + self.global_step / (0.5 * self.opt.iters))
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                    self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                    self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + self.global_step / (0.5 * self.opt.iters))

        rays_o = data['rays_o'] # [B, N, 3] # B,F,N,3
        rays_d = data['rays_d'] # [B, N, 3] # B,F,N,3
        mvp = data['mvp'] # [B, 4, 4] / [B,F,4,4]
        time = data['time'] # [B,T]

        use_dynamic_cam = (rays_o.ndim == 4)
        B = rays_o.size(0)
        # ipdb.set_trace()
        N = rays_o.size(1) if not use_dynamic_cam else rays_o.size(2)

        H, W = data['H'], data['W']

        # ipdb.set_trace()
        start_from_zero = data.get('start_from_zero', True)
        if start_from_zero:
            assert time[0,0] == 0
        

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False
            bg_color = self.get_bg_color(
                self.opt.bg_color_known, B*N, rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale #* (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif self.global_step < (self.opt.latent_iter_ratio * self.opt.iters): ## 0
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if self.global_step < (self.opt.normal_iter_ratio * self.opt.iters): # 0.2
                ambient_ratio = 1.0
                shading = 'normal'
            elif self.global_step < (self.opt.textureless_iter_ratio * self.opt.iters): # 0
                ambient_ratio = 0.1 + 0.9 * random.random()
                shading = 'textureless'
            elif self.global_step < (self.opt.albedo_iter_ratio * self.opt.iters): # 0
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = 0.1 + 0.9 * random.random()
                rand = random.random()
                if rand < self.opt.textureless_rate: # 0.2
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            # ipdb.set_trace()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg
            
            ## NOTE if bg_radius < 0 -> the way magic123 use
            # The bg color is always random

        video_outputs = []
        num_frames = time.size(1)

        light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device))
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, time=time, do_rgbd_loss=do_rgbd_loss, light_d=light_d)
        # ipdb.set_trace()

        pred_depth = outputs['depth'].reshape(B, num_frames, 1, H, W)
        if self.opt.normalize_depth: 
            pred_depth = nonzero_normalize_depth(pred_depth)
        pred_mask = outputs['weights_sum'].reshape(B, num_frames, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, num_frames, H, W, 3)
        else:
            pred_normal = None 

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, num_frames, H, W, 4).permute(0,1,4,2,3).contiguous() # [B, F, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, num_frames, H, W, 3).permute(0,1,4,2,3).contiguous() # [B, F, 3, H, W]

        # ipdb.set_trace()
        if 'image_wo_bg' in outputs:
            image_wo_bg = outputs['image_wo_bg'] + (1 - outputs['weights_sum']).unsqueeze(-1) * 1 # B,F,N,3
            if as_latent:
                # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
                pred_rgb_wobg = torch.cat([image_wo_bg, outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, num_frames, H, W, 4).permute(0,1,4,2,3).contiguous() # [B, 4, H, W]
            else:
                pred_rgb_wobg = image_wo_bg.reshape(B, num_frames, H, W, 3).permute(0,1,4,2,3).contiguous() # [B, 3, H, W]

        out_dict = {
            'rgb': pred_rgb, # B,F,3,H,W
            'depth': pred_depth, # B,F,1,H,W
            'mask': pred_mask, # B,F,1,H,W
            'normal': pred_normal, # B,F,H,W,3
            'pred_rgb_wobg': pred_rgb_wobg
        }

        # Loss
        # known view loss
        loss_rgb, loss_mask, loss_normal, loss_depth, loss_sds, loss_if, loss_zero123, loss_clip, loss_entropy, loss_opacity, loss_orient, loss_smooth, loss_smooth2d, loss_smooth3d, loss_mesh_normal, loss_mesh_lap, loss_time_tv, loss_canonical, loss_sr, loss_cn = torch.zeros(20, device=self.device)
        # known view loss
        # assert not do_rgbd_loss 
        reg_losses_dict = {}

        loss = 0

        if do_rgbd_loss:
            ## NOTE this only applied to the first frame, 
            # ipdb.set_trace()
            gt_mask = self.mask # [B, H, W] bool
            gt_rgb = self.rgb   # [B, 3, H, W]
            gt_opacity = self.opacity   # [B, 1, H, W] # float version of mask
            gt_normal = self.normal # [B, H, W, 3] # None
            gt_depth = self.depth   # N -> only mask true depth

            if len(gt_rgb) > self.opt.batch_size:
                gt_mask = gt_mask[choice]
                gt_rgb = gt_rgb[choice]
                gt_opacity = gt_opacity[choice]
                gt_normal = gt_normal[choice]
                gt_depth = gt_depth[choice]

            # color loss
            loss_rgb = self.opt.lambda_rgb * self.rgbd_scale * \
                F.mse_loss(pred_rgb[:,0]*gt_opacity, gt_rgb*gt_opacity) # B,3,H,W

            # mask loss
            loss_mask = self.opt.lambda_mask * self.rgbd_scale * F.mse_loss(pred_mask[:,0], gt_mask.to(torch.float32).unsqueeze(0))

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs and self.normal is not None:
                pred_normal = pred_normal[:,0][self.mask]
                lambda_normal = self.opt.lambda_normal * \
                    min(1, self.global_step / self.opt.iters)                
                loss_normal = lambda_normal * self.rgbd_scale * \
                    (1 - F.cosine_similarity(pred_normal, self.normal).mean())/2

            # relative depth loss
            if self.opt.lambda_depth > 0 and self.depth is not None:
                valid_pred_depth = pred_depth[:, 0, 0][self.mask]
                loss_depth = self.opt.lambda_depth * self.rgbd_scale * (1 - pearson_corrcoef(valid_pred_depth, self.depth))/2
            
            loss = (loss_rgb + loss_mask + loss_normal + loss_depth)
        # novel view loss
        else:
            # ipdb.set_trace()
            static_rgb = None

            save_guidance_path = os.path.join(self.opt.workspace, 'guidance', f'train_step{self.global_step}_guidance.jpg') if self.opt.save_guidance_every > 0 and self.global_step % self.opt.save_guidance_every ==0 else None
            if 'SD' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]
                # ipdb.set_trace()
                ## NOTE here should I remove the view information?
                ## add mid frame view information
                if 'frame_azimuth' in data and use_dynamic_cam:
                    idx = num_frames//2
                    azimuth = data['frame_azimuth'][idx:idx+1] # 1,3

                # ENHANCE: remove loop to handle batch size > 1
                text_z = [] 
                for b in range(azimuth.shape[0]):
                    if self.opt.no_view_text and use_dynamic_cam:
                        text_z.append(self.embeddings['SD']['default'])
                        continue

                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['SD']['front']
                        end_z = self.embeddings['SD']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['SD']['side']
                        end_z = self.embeddings['SD']['back']
                    text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.stack(text_z, dim=0).transpose(0, 1).flatten(0, 1)
                # text_z_sds = text_z[:, :-1]  # this is to remove the cls token...
                text_z_sds = text_z
                loss_sds, _ = self.guidance['SD'].train_step(text_z_sds, pred_rgb, as_latent=as_latent, guidance_scale=self.opt.guidance_scale['SD'], grad_scale=self.opt.lambda_guidance['SD'],
                                                             density=pred_mask if self.opt.gudiance_spatial_weighting else None, 
                                                             save_guidance_path=save_guidance_path,
                                                             step=self.global_step,
                                                             )


            if 'CN' in self.guidance:
                # ipdb.set_trace()
                save_guidance_CN_path = os.path.join(self.opt.workspace, 'guidance_CN', f'train_step{self.global_step}_guidance.jpg') if self.opt.save_guidance_every > 0 and self.global_step % self.opt.save_guidance_every ==0 else None
                # ipdb.set_trace()
                ## NOTE here should not use text_z_sds, if the SR model use different text encoder?

                ## get image index for part frames update
                index = torch.arange(0, self.opt.num_frames, step=1) # default, choose all
                if self.opt.cn_frames < self.opt.num_frames:
                    if self.opt.cn_frame_method == 'even':
                        assert self.opt.num_frames % self.opt.cn_frames == 0
                        interval = self.opt.num_frames // self.opt.cn_frames
                        index = torch.arange(0, self.opt.num_frames, step=interval)
                    elif self.opt.cn_frame_method == 'random':
                        index = torch.randperm(self.opt.num_frames)[:self.opt.cn_frames]
                    else:
                        raise NotImplementedError

                azimuth = data['azimuth'] # [-180, 180]
                # ipdb.set_trace()
                ## NOTE here should I remove the view information?
                if 'frame_azimuth' in data and use_dynamic_cam:
                    azimuth = data['frame_azimuth'][index] # N,3
                    
                # ENHANCE: remove loop to handle batch size > 1
                text_z = [] 
                for b in range(azimuth.shape[0]):
                    if self.opt.no_view_text and use_dynamic_cam:
                        text_z.append(self.embeddings['CN']['default'])
                        continue

                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['CN']['front']
                        end_z = self.embeddings['CN']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['CN']['side']
                        end_z = self.embeddings['CN']['back']
                    text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.stack(text_z, dim=0).transpose(0, 1).flatten(0, 1) # TODO check B,2,77,C -> 2B,77,C?
                # text_z_sds = text_z[:, :-1]  # this is to remove the cls token...
                text_cn_sds = text_z
                text_cn_cn = self.embeddings['CN']['CN'] if self.opt.cn_cn_text else text_cn_sds

                ## NOTE here we use online prediction, will this lead to error accumulation -> Yes
                ## get the condition images
                cn_cn_pred_rgb = pred_rgb.detach()
                cn_pred_rgb = pred_rgb
                # pred_rgb B,F,3,H,W
                # ipdb.set_trace()

                if self.fix_dynamic:
                    ## NOTE dynamic render is not applied to the inference model, so the render should be the training model
                    with torch.no_grad():
                        outputs_dyn = self.dynamic_model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, time=time, do_rgbd_loss=do_rgbd_loss, light_d=light_d)
                    # ipdb.set_trace()

                    pred_depth_dyn = outputs_dyn['depth'].reshape(B, num_frames, 1, H, W)
                    if self.opt.normalize_depth: 
                        pred_depth_dyn = nonzero_normalize_depth(pred_depth_dyn)

                    pred_rgb_dyn = outputs_dyn['image'].reshape(B, num_frames, H, W, 3).permute(0,1,4,2,3).contiguous() # [B, F, 3, H, W]
                    ## use the dynamic rendered from the fixed model for controlnet input
                    cn_cn_pred_rgb = pred_rgb_dyn

                ## select image with index
                cn_pred_rgb = pred_rgb[:,index]
                cn_cn_pred_rgb = cn_cn_pred_rgb[:,index]

                loss_cn, _ = self.guidance['CN'].train_step(text_cn_sds, text_cn_cn, cn_pred_rgb, cn_cn_pred_rgb, as_latent=as_latent, guidance_scale=self.opt.guidance_scale['CN'], grad_scale=self.opt.lambda_guidance['CN'], density=None, save_guidance_path=save_guidance_CN_path, step=self.global_step,)

            if 'IF' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]

                # ENHANCE: remove loop to handle batch size > 1
                # ENHANCE: remove loop to handle batch size > 1
                text_z = [] 
                for b in range(azimuth.shape[0]):
                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['IF']['front']
                        end_z = self.embeddings['IF']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['IF']['side']
                        end_z = self.embeddings['IF']['back']
                    text_z.append(r * start_z + (1 - r) * end_z)
                text_z = torch.stack(text_z, dim=0).transpose(0, 1).flatten(0, 1)
                text_z = torch.cat(text_z, dim=1).reshape(B, 2, start_z.shape[-2]-1, start_z.shape[-1]).transpose(0, 1).flatten(0, 1)
                loss_if = self.guidance['IF'].train_step(text_z, pred_rgb, guidance_scale=self.opt.guidance_scale['IF'], grad_scale=self.opt.lambda_guidance['IF'])

            if 'zero123' in self.guidance and start_from_zero:
                # raise NotImplementedError
                save_guidance_zero123_path = os.path.join(self.opt.workspace, 'guidance_zero123', f'train_step{self.global_step}_guidance.jpg') if self.opt.save_guidance_every > 0 and self.global_step % self.opt.save_guidance_every ==0 else None
                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                # input_3dprior B,3,H,W
                # ipdb.set_trace()
                input_3dprior = pred_rgb[:,0]

                loss_zero123 = self.guidance['zero123'].train_step(self.embeddings['zero123']['default'], input_3dprior, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale['zero123'],
                                                                  as_latent=as_latent, grad_scale=self.opt.lambda_guidance['zero123'], save_guidance_path=save_guidance_zero123_path)

            if 'clip' in self.guidance:

                # empirical, far view should apply smaller CLIP loss
                lambda_guidance = 10 * (1 - abs(azimuth) / 180) * self.opt.lambda_guidance['clip']
                loss_clip = self.guidance['clip'].train_step(self.embeddings['clip'], pred_rgb, grad_scale=lambda_guidance)
            loss += loss_sds + loss_if + loss_zero123 + loss_clip + loss_sr + loss_cn

        # regularizations
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0: # 0
                loss_opacity = self.opt.lambda_opacity * (outputs['weights_sum'] ** 2).mean()
                reg_losses_dict['loss_opacity'] = loss_opacity.item()

            if self.opt.lambda_entropy > 0: # 1e-3
                lambda_entropy = self.opt.lambda_entropy * \
                    min(1, 2 * self.global_step / self.opt.iters)
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = lambda_entropy * (- alphas * torch.log2(alphas) -
                                (1 - alphas) * torch.log2(1 - alphas)).mean()
                reg_losses_dict['loss_entropy'] = loss_entropy.item()

            if self.opt.lambda_normal_smooth > 0 and 'normal_image' in outputs: # 0.5 # no image in sd-dreamfusion should be 0
                pred_vals = outputs['normal_image'].reshape(-1, H, W, 3) # BF,H,W,3
                # total-variation
                loss_smooth = (pred_vals[:, 1:, :, :] - pred_vals[:, :-1, :, :]).square().mean() + \
                              (pred_vals[:, :, 1:, :] -
                               pred_vals[:, :, :-1, :]).square().mean()
                loss_smooth = self.opt.lambda_normal_smooth * loss_smooth
                reg_losses_dict['loss_smooth'] = loss_smooth.item()

            if self.opt.lambda_normal_smooth2d > 0 and 'normal_image' in outputs: # 0.5 # no image in sd-dreamfusion should be 0
                pred_vals = outputs['normal_image'].reshape(
                    -1, H, W, 3).permute(0,3,1,2).contiguous() # BF,3,H,W
                smoothed_vals = TF.gaussian_blur(pred_vals, kernel_size=9)
                loss_smooth2d = self.opt.lambda_normal_smooth2d * F.mse_loss(pred_vals, smoothed_vals)
                reg_losses_dict['loss_smooth2d'] = loss_smooth2d.item()

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs: # 1e-2
                loss_orient = self.opt.lambda_orient * outputs['loss_orient'].mean()
                reg_losses_dict['loss_orient'] = loss_orient.item()
            
            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs: # 0
                loss_smooth3d = self.opt.lambda_3d_normal_smooth * outputs['loss_normal_perturb'].mean()
                reg_losses_dict['loss_smooth3d'] = loss_smooth3d.item()
            
            if self.opt.lambda_time_tv > 0:
                if self.opt.backbone == 'grid4d': 
                    loss_time_tv = self.opt.lambda_time_tv * self.model.TV_loss()
                    reg_losses_dict['loss_time_tv'] = loss_time_tv.item()


            loss += loss_opacity + loss_entropy + loss_smooth + loss_smooth2d + loss_orient + loss_smooth3d + loss_time_tv + loss_canonical
            
        else:
            if self.opt.lambda_mesh_normal > 0:
                loss_mesh_normal = self.opt.lambda_mesh_normal * \
                    outputs['loss_normal'].mean()
                reg_losses_dict['loss_mesh_normal'] = loss_mesh_normal.item()

            if self.opt.lambda_mesh_lap > 0:
                loss_mesh_lap = self.opt.lambda_mesh_lap * outputs['loss_lap'].mean()
                reg_losses_dict['loss_mesh_lap'] = loss_mesh_lap.item()

            loss += loss_mesh_normal + loss_mesh_lap

            
        losses_dict = {
                    'loss': loss.item(),
                    'loss_sds': loss_sds.item(),
                    'loss_sr': loss_sr.item(),
                    'loss_cn': loss_cn.item(),
                    # 'loss_if': loss_if.item(),
                    'loss_zero123': loss_zero123.item(),
                    # 'loss_clip': loss_clip.item(),
                    'loss_rgb': loss_rgb.item(),
                    'loss_mask': loss_mask.item(),
                    'loss_normal': loss_normal.item(),
                    'loss_depth': loss_depth.item(),
                    # 'loss_opacity': loss_opacity.item(),
                    # 'loss_entropy': loss_entropy.item(),
                    # 'loss_smooth': loss_smooth.item(),
                    # 'loss_smooth2d': loss_smooth2d.item(),
                    # 'loss_smooth3d': loss_smooth3d.item(),
                    # 'loss_orient': loss_orient.item(),
                    # 'loss_mesh_normal': loss_mesh_normal.item(),
                    # 'loss_mesh_lap': loss_mesh_lap.item(),
                }
        losses_dict.update(reg_losses_dict)
        # if loss_guidance_dict:
        #     for key, val in loss_guidance_dict.items():
        #         losses_dict[key] = val.item() if isinstance(val, torch.Tensor) else val
            
        if 'normal' in out_dict:
            out_dict['normal'] =  rearrange(out_dict['normal'], "b f h w c -> b f c h w").contiguous() # B,F,H,W,3 -> B,F,3,H,W

        if torch.isnan(loss):
            ipdb.set_trace()
            
        # save for debug purpose
        if self.opt.save_train_every > 0 and self.global_step % self.opt.save_train_every == 0:
            image_save_path = os.path.join(self.workspace, 'train_debug',)
            os.makedirs(image_save_path, exist_ok=True)
            for key, value in out_dict.items():
                if value is not None:
                    value = ((value - value.min()) / (value.max() - value.min() + 1e-6)).detach().mul(255).to(torch.uint8) # 0-255
                    try:
                        save_tensor2image(value, os.path.join(image_save_path, f'train_{self.global_step:06d}_{key}.jpg'), channel_last=False) 
                    except:
                        pass
        return loss, losses_dict, out_dict 


    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)


        if not self.opt.dmtet and (self.opt.backbone in ['grid4d']):
            # ipdb.set_trace()
            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0: ## not implement
                self.model.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'lambertian' 
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        time = data['time']
        num_frames = time.size(1)
        ## time should be batch size 1
        assert time.size(0) == 1
        video_outputs = []


        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, time=time)

        pred_rgb = outputs['image'].reshape(B, num_frames, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, num_frames, H, W, 1)
        if self.opt.normalize_depth: 
            pred_depth = nonzero_normalize_depth(pred_depth)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, num_frames, H, W, 3)
        else:
            pred_normal = None 
        out_dict = {
            shading: pred_rgb,
            'depth': pred_depth,
            'normal_image': pred_normal,
        }
        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)
        return out_dict, loss

    def test_step(self, data, bg_color=None, perturb=False, shading='lambertian'):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        bg_color = self.get_bg_color(bg_color, B*N, rays_o.device)

        shading = data['shading'] if 'shading' in data else shading 
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        ## during eval and test, ambient_ratio is always 1 -> then the rendered image is just albedo

        time = data['time']
        num_frames = time.size(1)
        ## time should be batch size 1
        assert time.size(0) == 1

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color, time=time)

        pred_rgb = outputs['image'].reshape(B, num_frames, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, num_frames, H, W, 1)
        pred_mask = outputs['weights_sum'].reshape(B, num_frames, H, W, 1)
        # if self.opt.normalize_depth: 
        pred_depth = nonzero_normalize_depth(pred_depth)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, num_frames, H, W, 3)
            pred_normal = pred_normal * pred_mask + (1.0 - pred_mask) 
        else:
            pred_normal = None 
        out_dict = {
            shading: pred_rgb,
            'depth': pred_depth,
            'normal_image': pred_normal,
            'mask': pred_mask,
        }
        return out_dict


