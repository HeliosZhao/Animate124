import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb

from nerf.utils import get_rays, safe_normalize
from nerf.provider import NeRFDataset, visualize_poses, DIR_COLORS, get_view_direction, rand_poses, circle_poses, generate_grid_points
import logging

logger = logging.getLogger(__name__)

def rand_poses_trajectory(size, device, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, static_view_rate=0.):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius [1.8,1.8]
        theta_range: [min, max], should be in [45, 135]
        phi_range: [min, max], should be in [-180, 180]
    Return:
        poses: [size, 4, 4]
    '''
    assert size == 1 and not opt.jitter_pose
    
    theta_range = np.array(theta_range) / 180 * np.pi # -pi/4 ~ 3pi/4
    phi_range = np.array(phi_range) / 180 * np.pi # -pi ~ pi
    angle_overhead = angle_overhead / 180 * np.pi # pi/6
    angle_front = angle_front / 180 * np.pi # pi/3

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate: # 0.5
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        init_phis = phis # init phi can be smaller than 0
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0] # 1
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        init_phis = phis
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0
    # scale_delta = False
    # ipdb.set_trace()
    ## delta thetas
    d_theta_range = [-np.pi/4, np.pi/4]
    d_phi_range = [-np.pi/2, np.pi/2]
    d_thetas = torch.rand(size, device=device) * (d_theta_range[1] - d_theta_range[0]) + d_theta_range[0] # -np.pi/4, np.pi/4
    d_phis = torch.rand(size, device=device) * (d_phi_range[1] - d_phi_range[0]) + d_phi_range[0] # -np.pi/2, np.pi/2
    if opt.scale_delta:
        ## scale delta will make the camera not exceed the range, also, it cannot across the range
        # for example, phi from -pi/4 to pi/4 is a reasonable motion but scale delta will make it impossible
        d_thetas = d_thetas.clamp(theta_range[0]-thetas, theta_range[1]-thetas) # d_theta + theta in range [theta_range[0], theta_range[1]]
        d_phis = d_phis.clamp(phi_range[0]-init_phis, phi_range[1]-init_phis) # d_phi + init_phi in range [phi_range[0], phi_range[1]] # init phi is before convert to 0-2pi

    ## 
    num_frames = opt.num_frames
    scale = torch.arange(num_frames, device=device) / num_frames # 0,1/f, ... f-1/f, F
    thetas_dyn = thetas + scale * d_thetas # F
    phis_dyn = init_phis + scale * d_phis # F
    phis_dyn[phis_dyn < 0] += 2 * np.pi
    assert thetas_dyn[0] == thetas[0] and phis_dyn[0] == init_phis[0]

    centers = torch.stack([
        radius * torch.sin(thetas_dyn) * torch.sin(phis_dyn),
        radius * torch.cos(thetas_dyn),
        radius * torch.sin(thetas_dyn) * torch.cos(phis_dyn),
    ], dim=-1) # [B, 3] # F,3

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(num_frames, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas_dyn, phis_dyn, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas_dyn = thetas_dyn / np.pi * 180
    phis_dyn = phis_dyn / np.pi * 180
    radius = radius.repeat(num_frames)

    return poses, dirs, thetas_dyn, phis_dyn, radius



class DNeRFDataset(NeRFDataset):
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__(opt, device, type, H, W, size)
        self.num_frames = opt.num_frames
        self.num_test_frames = opt.get("num_test_frames", self.num_frames)
        self.dynamic_cam_rate = self.opt.dynamic_cam_rate
        self.precision = opt.get('precision', 64)
        self.zero_precision = opt.get('zero_precision', self.precision)
        logger.info(f"Training dataset, random time sampling precision is {self.precision}, zero time sampling precision is {self.zero_precision}")

        
    def get_default_view_data(self):

        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx = H / 2
        cy = W / 2

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device)
        thetas = torch.FloatTensor(self.opt.ref_polars).to(self.device)
        phis = torch.FloatTensor(self.opt.ref_azimuths).to(self.device)
        poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        projection = torch.tensor([
            [2*focal/W, 0, 0, 0],
            [0, -2*focal/H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(radii), 1, 1)

        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, H, W, -1)

        if rays['rays_o'].size(0):
            time = torch.FloatTensor([0]).reshape(rays['rays_o'].size(0), 1)
        else:
            time = None

        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'time': time,
            'polar': self.opt.ref_polars,
            'azimuth': self.opt.ref_azimuths,
            'radius': self.opt.ref_radii,
        }

        return data

    def collate(self, index):

        B = len(index)

        dynamic_cam = False
        start_from_zero = False
        if self.training:
            if np.random.random() < self.dynamic_cam_rate:
                dynamic_cam = True
                poses, dirs, thetas, phis, radius = rand_poses_trajectory(B, self.device, self.opt, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, uniform_sphere_rate=self.opt.uniform_sphere_rate)
                ## poses F,4,4
            else:
                # random pose on the fly, size 1,4,4
                poses, dirs, thetas, phis, radius = rand_poses(B, self.device, self.opt, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]

            zero_precision = self.zero_precision
            zero_rate = self.opt.zero_rate
            full_precision = self.precision
            num_frames = self.num_frames

            start_from_zero = False
            precision = full_precision
            if random.random() < zero_rate:
                start_from_zero = True
                precision = zero_precision

            frame_time = int( random.randint(num_frames, precision) / num_frames ) * num_frames
            #  16, 32, 48, 64
            time_interval = int(frame_time/num_frames) # 1,2,3,4
            ## sample zero rate
            if start_from_zero:
                start_time = 0
            else:
                # start_time = random.uniform(0.0, 1.0 - frame_time)
                start_time = random.randint(0, precision-frame_time) # select one from 64-N

            # time_interval = torch.arange(self.num_frames) / self.num_frames
            # time = start_time + time_interval * frame_time
            time = ((np.arange(num_frames) * time_interval) + start_time) / precision * full_precision

            if random.random() > 0.5:
                # ipdb.set_trace()
                time = full_precision - time
                time = time.tolist()
                time = np.array(time[::-1])
                start_from_zero = False

            # ipdb.set_trace()
            time = torch.from_numpy(time).float() / self.precision
            assert time.max() <= 1 and time.min()>=0 and time.size(0) == self.num_frames


        elif self.type == 'six_views':
            raise NotImplementedError
            # six views
            thetas_six = [90]*4 + [1e-6] + [180]
            phis_six = [0, 90, 180, -90, 0, 0]
            thetas = torch.FloatTensor([thetas_six[index[0]]]).to(self.device)
            phis = torch.FloatTensor([phis_six[index[0]]]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = self.opt.default_fovy

        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.default_polar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            phis = phis + self.opt.default_azimuth
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = self.opt.default_fovy
            time = torch.arange(self.num_test_frames) / self.num_test_frames

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        # ipdb.set_trace()
        rays = get_rays(poses, intrinsics, self.H, self.W, -1) # F,N,3

        # delta polar/azimuth/radius to default view
        ## NOTE apply to the first view
        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius

        if dynamic_cam:
            data_rays_o = rays['rays_o'][None] # 1,F,N,3
            data_rays_d = rays['rays_d'][None] # 1,F,N,3
            mvp = mvp[None] # 1,F,4,4
            dirs = dirs[None]
        
        else:
            data_rays_o = rays['rays_o'] # 1,N,3
            data_rays_d = rays['rays_d'] # 1,N,3

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': data_rays_o, # 1,HW,3
            'rays_d': data_rays_d, # 1,HW,3
            'time': time[None], # B,T
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar[:1],
            'azimuth': delta_azimuth[:1],
            'radius': delta_radius[:1],
            'frame_azimuth': delta_azimuth, # F,3
            'start_from_zero': start_from_zero,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader
