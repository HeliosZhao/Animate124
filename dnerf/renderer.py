import os
import math
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


import raymarching
from nerf.utils import custom_meshgrid, safe_normalize
from nerf.renderer import NeRFRenderer
import logging


logger = logging.getLogger(__name__)


class DNeRFRenderer(NeRFRenderer):
    def __init__(self, opt):
        super().__init__(opt)

        self.time_size = opt.get("time_size", 1)
        self.density_scale = opt.get("density_scale", 1)
        self.dynamic_ft = opt.get("dynamic_ft", False)

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid (with an extra time dimension)
            density_grid = torch.zeros(self.time_size, self.cascade, self.grid_size ** 3) # [T, CAS, H * H * H]
            density_bitfield = torch.zeros(self.time_size, self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [T, CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # time stamps for density grid
            times = ((torch.arange(self.time_size, dtype=torch.float32) + 0.5) / self.time_size).view(-1, 1, 1) # [T, 1, 1]
            self.register_buffer('times', times)
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0


    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, time=None, **kwargs):
        # rays_o, rays_d: [B, N, 3] / B,F,N,3
        # return: image: [B, N, 3], depth: [B, N]
        # time: [B,F]

        prefix = rays_o.shape[:-1]
        batch_size = prefix[0]
        if prefix[0] != 1:
            raise "The prefix should be 1 if different frames has different camera pose in the current version"
    

        dynamic_cam = True if rays_o.ndim == 4 else False



        N = rays_o.shape[:-1].numel() # B * N, in fact
        device = rays_o.device

        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [B,N,3] / B,F,N,3

        if time is None:
            assert not self.dynamic_ft
            time_steps = torch.LongTensor([[0]]).reshape(1,1) # 1,1
            time = torch.FloatTensor([[0]]).reshape(1,1) # 1,1
            num_frames = 1
        else:
            time_steps = torch.floor(time * self.time_size).clamp(min=0, max=self.time_size - 1).long() # B,F
            num_frames = time.size(1)
        
        if dynamic_cam:
            rays_o = rays_o[0].contiguous() # F,N,3
            rays_d = rays_d[0].contiguous() # F,N,3
            light_d = light_d[0].contiguous() # F,N,3
        else:
            rays_o = rays_o.repeat(num_frames, 1, 1).contiguous()
            rays_d = rays_d.repeat(num_frames, 1, 1).contiguous()
            light_d = light_d.repeat(num_frames, 1, 1).contiguous()
            # ipdb.set_trace()

        

        results = {}
        # ipdb.set_trace()
        if self.training:
            # ipdb.set_trace()
            v_xyzs = []
            v_dirs = []
            v_light = []
            v_time = []
            v_idx = [0]
            v_rays = []
            v_ts = []
            v_kernels = []
            for frame_idx in range(num_frames):
                _rays_o, _rays_d, _light_d, t = rays_o[frame_idx], rays_d[frame_idx], light_d[frame_idx], time_steps[0,frame_idx].item() 
                ## N,3 for the first 3, t is a value
                # pre-calculate near far
                nears, fars = raymarching.near_far_from_aabb(_rays_o, _rays_d, self.aabb_train if self.training else self.aabb_infer)
                xyzs, dirs, ts, rays = raymarching.march_rays_train(_rays_o, _rays_d, self.bound, self.density_bitfield[t], self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)
                dirs = safe_normalize(dirs)

                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                if _light_d.shape[0] > 1:
                    _light_d = _light_d[flatten_rays]
                else: # 1,3
                    _light_d = _light_d.repeat(xyzs.size(0), 1)
                
                v_xyzs.append(xyzs)
                v_dirs.append(dirs)
                v_light.append(_light_d)
                sample_num = xyzs.size(0)

                v_time.append(torch.zeros((sample_num,1), device=xyzs.device, dtype=xyzs.dtype)+time[0, frame_idx].item()) # NOTE this should be real time # N
                v_idx.append(sample_num)

                v_rays.append(rays)
                v_ts.append(ts)
            
            v_xyzs = torch.cat(v_xyzs, dim=0)
            v_dirs = torch.cat(v_dirs, dim=0)
            v_light = torch.cat(v_light, dim=0)
            v_time = torch.cat(v_time, dim=0)
            v_idx = np.cumsum(v_idx).tolist()
            
            sigmas, rgbs, normals, deforms = self(v_xyzs, v_dirs, v_light, ratio=ambient_ratio, shading=shading, t=v_time)
    
            sigmas = self.density_scale * sigmas
            weights = []
            weights_sum = []
            depth = []
            image = []
            normal_image = []




            # ipdb.set_trace()
            for frame_idx in range(num_frames):
                start_idx, end_idx = v_idx[frame_idx], v_idx[frame_idx+1] 
                # _weights, _weights_sum, _depth, _image
                out = raymarching.composite_rays_train(sigmas[start_idx:end_idx], rgbs[start_idx:end_idx], v_ts[frame_idx], v_rays[frame_idx], T_thresh, binarize)
                if normals is not None:
                    _, _, _, _normal_image = raymarching.composite_rays_train(sigmas[start_idx:end_idx].detach(), (normals[start_idx:end_idx] + 1) / 2, v_ts[frame_idx], v_rays[frame_idx], T_thresh, binarize)
                    normal_image.append(_normal_image)

                weights.append(out[0])
                weights_sum.append(out[1])
                depth.append(out[2])
                image.append(out[3]) # N,3
            
            weights = torch.cat(weights)
            weights_sum = torch.stack(weights_sum, dim=0) # F,N
            depth = torch.stack(depth, dim=0) # F,N
            image = torch.stack(image, dim=0) # F,N,3
            
            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * v_dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(v_xyzs + torch.randn_like(v_xyzs) * 1e-2, t=v_time)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()

            
            if normals is not None:
                # _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize)
                # results['normal_image'] = normal_image.view(*prefix, 3)
                normal_image = torch.stack(normal_image, dim=0) # F,N,3
                results['normal_image'] = normal_image.view(batch_size, num_frames, -1, 3) # B,F,N,3
            
            # weights normalization
            results['weights'] = weights # N'*F

        else:
            image_all = []
            weights_sum_all = []
            depth_all = []
            # ipdb.set_trace()
            for frame_idx in range(num_frames): 
                _rays_o, _rays_d, _light_d, t = rays_o[frame_idx], rays_d[frame_idx], light_d[frame_idx], time_steps[0,frame_idx].item() 
                nears, fars = raymarching.near_far_from_aabb(_rays_o, _rays_d, self.aabb_train if self.training else self.aabb_infer)
                # allocate outputs 
                dtype = torch.float32
                
                weights_sum = torch.zeros(N, dtype=dtype, device=device)
                depth = torch.zeros(N, dtype=dtype, device=device)
                image = torch.zeros(N, 3, dtype=dtype, device=device)
                
                n_alive = N
                rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
                rays_t = nears.clone() # [N] ## test must use the same 

                step = 0
                
                while step < self.opt.max_steps: # hard coded max step

                    # count alive rays 
                    n_alive = rays_alive.shape[0]

                    # exit loop
                    if n_alive <= 0:
                        break

                    # decide compact_steps
                    n_step = max(min(N // n_alive, 8), 1)

                    xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, _rays_o, _rays_d, self.bound, self.density_bitfield[t], self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                    dirs = safe_normalize(dirs)

                    s_time = torch.zeros((xyzs.size(0),1), device=xyzs.device, dtype=xyzs.dtype) + time[0, frame_idx].item()
                    # ipdb.set_trace()
                    sigmas, rgbs, normals, _ = self(xyzs, dirs, _light_d[:1], ratio=ambient_ratio, shading=shading, t=s_time)
                    sigmas = self.density_scale * sigmas
                    raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)

                    rays_alive = rays_alive[rays_alive >= 0]
                    #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                    step += n_step

                image_all.append(image)
                weights_sum_all.append(weights_sum)
                depth_all.append(depth)
            # ipdb.set_trace()
            weights_sum = torch.stack(weights_sum_all, dim=0) # F,N
            depth = torch.stack(depth_all, dim=0) # F,N
            image = torch.stack(image_all, dim=0) # F,N,3

        # mix background color
        ## when bg_radius < 0 -> the way Magic123, during training, bg_color is always a random color, during inference, always 1 
        # ipdb.set_trace()
        if bg_color is None:
            if self.opt.bg_radius > 0 and self.bg_net is not None:
                # use the bg model to calculate bg_color
                ## NOTE here the camera should be fixed in the video
                ## rays_d F,N,3
                bg_color = self.background(rays_d.reshape(-1, 3)) # [FN, 3] # this is irrelavant to time
                bg_color = bg_color.reshape(batch_size, num_frames, -1, 3) # F,N,3
            else:
                bg_color = 1

        image_wo_bg = image.view(batch_size, num_frames, -1, 3)
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(batch_size, num_frames, -1, 3)

        depth = depth.view(batch_size, num_frames, -1)

        weights_sum = weights_sum.reshape(batch_size, num_frames, -1)

        results['image'] = image # B,F,N,3
        results['depth'] = depth # B,F,N
        results['weights_sum'] = weights_sum # B,F,N
        results['image_wo_bg'] = image_wo_bg # B,F,N,3
        # ipdb.set_trace()
        return results



    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not (self.cuda_ray):
            return 
        if self.taichi_ray:
            raise NotImplementedError
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        # if self.iter_density < 16: # update only 16 times
        if True: # full update
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for t, time in enumerate(self.times):
                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                half_time_size = 0.5 / self.time_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in coord [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # add noise in time [-hts, hts]
                                time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                                # query density
                                sigmas = self.density(cas_xyzs, time_perturb)['sigma'].reshape(-1).detach()
                                sigmas *= self.density_scale
                                # assign 
                                tmp_grid[t, cas, indices] = sigmas


        # ema update in magic123
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        for t in range(self.time_size):
            self.density_bitfield[t] = raymarching.packbits(self.density_grid[t], density_thresh, self.density_bitfield[t])

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0


