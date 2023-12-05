import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import DNeRFRenderer as NeRFRenderer

import numpy as np
from encoding import get_encoder

from nerf.utils import safe_normalize
import ipdb
import copy
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 encoding_time="frequency",
                 encoding_deform="frequency", # "hashgrid" seems worse
                 num_layers=2,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=3, # a deeper MLP is very necessary for performance. # before 5
                 hidden_dim_deform=32, # before 128
                 level_dim=2,
                 bound=1,
                 ):
        
        super().__init__(opt)
        self.level_dim = opt.level_dim if hasattr(opt, 'level_dim') else level_dim
        self.dynamic = opt.get('dynamic', False)
        self.dynamic_ft = opt.get('dynamic_ft', False)
        self.freeze_first = opt.get('freeze_first', False)  # freeze the first encoder -> static

        self.lr_encoder_scale = opt.get('lr_encoder_scale', 10.0)
        self.lr_deform_scale = opt.get('lr_deform_scale', 1.0)
        self.lr_sigma_scale = opt.get('lr_sigma_scale', 1.0)

        self.num_chunk = opt.get('num_chunk', 1)
        self.use_color_net = opt.color_net

        num_levels = opt.get('num_levels', 16) ## num of levels for multi-reso
        num_layers = opt.get('num_layers', num_layers) ## num of layers
        hidden_dim = opt.get('hidden_dim', hidden_dim) ## num of layers

        self.time_grid_size = opt.get('time_grid_size', 1) # number of fusion head
        if not self.dynamic:
            self.time_grid_size = 1

        self.call_counter = 0
    
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        if opt.grid_type == 'tiledgrid' or opt.grid_type == 'tiledgrid_triplane':
            logging.info(f"This function should use tiledgrid and tiledgrid_triplane. Currently using {opt.grid_type}")
            encoder_params = {
                "encoding":opt.grid_type,
                "input_dim": 3,
                "level_dim": self.level_dim,
                "log2_hashmap_size": 16,
                "num_levels":num_levels,
                "desired_resolution": 2048 * self.bound,
            }

        elif opt.grid_type == 'hashgrid' or opt.grid_type == 'hashgrid_triplane':
            logging.info(f"This function should use hashgrid and hashgrid_triplane. Currently using {opt.grid_type}")

            encoder_params = {
                "encoding":opt.grid_type,
                "input_dim": 3,
                "level_dim": self.level_dim,
                "log2_hashmap_size": 19,
                "num_levels":num_levels,
                "desired_resolution": 2048 * self.bound,
                "interpolation":'smoothstep'
            }


        ## encoder state dict
        ## encoder.0.embeddings / encoder.0.offsets
        self.encoder = nn.ModuleList()
        for _ in range(self.time_grid_size):
            per_encoder, self.in_dim = get_encoder(
                **encoder_params
            )
            self.encoder.append(per_encoder)
        
        if self.dynamic_ft and self.freeze_first:
            for p in self.encoder[0].parameters():
                p.requires_grad = False



        ## NOTE here use stable dreamfusion instant ngp method, without color net
        sigma_in_dim = self.in_dim
        sigma_out_dim = 1 if self.use_color_net else 4

        self.color_net = None
        if self.use_color_net:
            self.sigma_net = MLP(sigma_in_dim, sigma_out_dim, hidden_dim, num_layers, bias=True)
            self.color_net = MLP(sigma_in_dim, 3, hidden_dim, num_layers, bias=True)

        else:
            self.sigma_net = MLP(sigma_in_dim, sigma_out_dim, hidden_dim, num_layers, bias=True)
        # self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)

        # masking
        self.grid_levels_mask = 0 

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None


    def common_forward_func(self, x, t=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        x = x.contiguous()
        deform = torch.zeros_like(x)
        time_feat = None

        if not self.dynamic:
            assert len(self.encoder) == 1
            h = self.encoder[0](x, bound=self.bound)

        else:
            # ipdb.set_trace()
            assert t is not None
            if t.size(0) == 1:
                t = t.repeat(x.size(0), 1) # N,1
            t = t.contiguous()

            ## NOTE for int value, ceil and floor are the same, so the dist should be dfloor and 1-dfloor
            ## t -> from 0 to 1
            ## NOTE the distance should be N-1, since 0 is always 0
            grid_idx = t * (self.time_grid_size - 1) # from floor to ceil
            grid_idx_ceil = grid_idx.ceil().int()
            grid_idx_floor = grid_idx.floor().int()
            x2 = x.repeat(2,1) # 2N,1
            grid_t = torch.cat([grid_idx_ceil, grid_idx_floor], dim=0)[...,0] # 2N,1 -> 2N
            h = torch.zeros((x2.size(0), self.in_dim), device=x2.device, dtype=x2.dtype)

            for tidx in range(self.time_grid_size):
                tmask = grid_t == tidx
                if tmask.sum():
                    cur_h = self.encoder[tidx](x2[tmask], bound=self.bound)
                    h = h.to(cur_h.dtype)
                    h[tmask] = cur_h
            h_ceil, h_floor = h.chunk(2, dim=0)
            dist_floor = grid_idx - grid_idx_floor # 0-1 dist to floor is the weight to ceil, more close to floor, the weight to ceil is smaller
            assert (dist_floor <= 1).all() and (dist_floor >= 0).all()
            h = h_ceil * dist_floor + h_floor * (1-dist_floor)


        # Feature masking for coarse-to-fine training
        if self.grid_levels_mask > 0:
            h_mask: torch.Tensor = torch.arange(self.in_dim, device=h.device) < self.in_dim - self.grid_levels_mask * self.level_dim  # (self.in_dim)
            h_mask = h_mask.reshape(1, self.in_dim).float()  # (1, self.in_dim)
            h = h * h_mask  # (N, self.in_dim)
        # ipdb.set_trace()
        sigma_h = self.sigma_net(h)

        #sigma = F.relu(h[..., 0])
        ## here is stable dreamfusion, no color net and add density blob
        sigma = self.density_activation(sigma_h[..., 0] + self.density_blob(x))

        if self.use_color_net:
            color_h = self.color_net(h)
            albedo = torch.sigmoid(color_h)
        
        else:
            albedo = torch.sigmoid(sigma_h[..., 1:])

        return sigma, albedo, deform
    
    def common_forward(self, x, t=None):
        # ipdb.set_trace()
        num_sample = x.size(0)
        chunk_size = num_sample // self.num_chunk # K
        if t is None:
            t = torch.zeros_like(x)[:,:1] # N,1
        elif t.size(0) != num_sample:
            t = t.repeat(num_sample, 1)

        sigma, albedo, deform = [], [], []
        # x_check = []
        for i in range(self.num_chunk):
            start = int(i*chunk_size)
            end = int((i+1)*chunk_size) if i != (self.num_chunk - 1) else num_sample
            
            _sig, _alb, _deform = self.common_forward_func(
                x[start: end],
                t[start: end]
                )
            # x_check.append(x[start: end])
            sigma.append(_sig)
            albedo.append(_alb)
            deform.append(_deform)
        
        # ipdb.set_trace()
        sigma = torch.cat(sigma, dim=0)
        albedo = torch.cat(albedo, dim=0)
        deform = torch.cat(deform, dim=0)
        assert sigma.size(0) == albedo.size(0) == num_sample == deform.size(0)

        return sigma, albedo, deform

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, t, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound), t)
        dx_neg, _, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound), t)
        dy_pos, _, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound), t)
        dy_neg, _, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound), t)
        dz_pos, _, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound), t)
        dz_neg, _, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound), t)
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x, t):
        
        normal = self.finite_difference_normal(x, t)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo', t=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        # assert t is not None
        sigma, albedo, deform = self.common_forward(x, t)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading
            # ipdb.set_trace()
            # normal = self.normal_net(enc)
            normal = self.normal(x, t)

            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                # ipdb.set_trace()
                if self.use_color_net:
                    color = albedo
                else:
                    color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal, deform

      
    def density(self, x, t=None):
        # x: [N, 3], in [-bound, bound]
        # assert t is not None
        sigma, albedo, _ = self.common_forward(x, t)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    @torch.no_grad()
    def density_blob(self, x):
        ## NOTE TODO in text to video, this is changed, the sigma is not fixed but gradually increase during training
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)

        if self.opt.density_activation == 'exp':
            g = self.opt.blob_density * \
                torch.exp(- d / (2 * self.opt.blob_radius ** 2))
        else:
            g = self.opt.blob_density * \
                (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g


    def background(self, d):
        # ipdb.set_trace()
        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs
 

    def get_optimvars(self, lr_static, lr_sigma, lr_dynamic):
        logger.info(f"Training with the following learning rate: Encoder {lr_static} \t Sigma net {lr_sigma} \t Deform net {lr_dynamic}")

        params = [
                    {'params': self.encoder.parameters(), 'lr': lr_static},
                    {'params': self.sigma_net.parameters(), 'lr': lr_sigma},
                ]


        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_sigma})
        
        return params

    # optimizer utils
    def get_params(self, lr):
        ## dynamic, the static grid encoder set to a low lr
        params = self.get_optimvars(lr*self.lr_encoder_scale, lr*self.lr_sigma_scale, lr*self.lr_deform_scale)

        return params

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.
        for tplane in self.encoder:
            # ipdb.set_trace()
            if tplane.embeddings.grad is not None:
                tplane.grad_total_variation(weight, inputs, bound, B)

    def TV_loss(self, ):
        # ipdb.set_trace()
        num_time_size = self.time_grid_size - 1
        total = 0
        for i in range(num_time_size):
            x1 = self.encoder[i].embeddings # N,2
            x2 = self.encoder[i+1].embeddings # N,2
            total = total + ((x1 - x2)**2).sum(-1).mean()

        total = total / (num_time_size * self.encoder[0].num_levels)
        
        return total


    def from_pretrained_3d(self, checkpoint_dict):
        ckpt = checkpoint_dict['model']
        new_model_state_dict = copy.deepcopy(self.state_dict())
        for k, v in self.state_dict().items():
            if k in ckpt:
                ckpt_v = ckpt[k]
                if (v.size() == ckpt_v.size()):
                    new_model_state_dict[k] = ckpt_v
                else:
                    # ipdb.set_trace()
                    ## NOTE check the density field, density grid and density bit field here are updated
                    logger.info("[INFO] Parameter {} has different sizes in checkpoint and model \n Size {} in the checkpoint and Size {} in the model".format(k, ckpt_v.size(), v.size()))
                    ## sigma net, load ckpt v as the first in dims layers and others as zero
                    ## sigma_net.net.0.weight: 64, 111 / 64,32
                    dynamic_weight = torch.zeros_like(v) # 111,64
                    dynamic_weight[:, :ckpt_v.size(1)] = ckpt_v
                    new_model_state_dict[k] = dynamic_weight
            else:
                assert 'encoder' in k
                ## encoder.x.embeddings and encoder.x.offsets
                layer_type = k.split('.')[-1] # embeddings or offsets
                ## temporal layers init as zero
                logger.info("[INFO] Parameter {} is not in the checkpoint, load from encoder.0.{}".format(k, layer_type))
                new_model_state_dict[k] = ckpt['encoder.0.{}'.format(layer_type)]
        
        self.load_state_dict(new_model_state_dict)

        if self.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.mean_density = checkpoint_dict['mean_density']


                    


