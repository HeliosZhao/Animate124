import torch
import argparse
import sys
import os
import pandas as pd
from omegaconf import OmegaConf

from nerf.provider import NeRFDataset, generate_grid_points
from nerf.utils import *

import yaml
from easydict import EasyDict as edict
import dnnultis
import logging


logger = logging.getLogger(__name__)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='3D AIGC Training')
parser.add_argument('--workspace', type=str, default='', help='path to log')
parser.add_argument('--text', default=None, help="text prompt")
parser.add_argument('--negative', default='', type=str,
                    help="negative text prompt")
parser.add_argument('--dir_texts_neg', action='store_true',
                    help="enable negative directional text")
parser.add_argument('--check_prompt', action='store_true', help="check prompt")

### model and training set up
parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
parser.add_argument('-O2', action='store_true',
                    help="equals --backbone vanilla")
parser.add_argument('-D', action='store_true',
                    help="dynamic nerf equals --backbone hexplane --fp16")
parser.add_argument('-DG', action='store_true',
                    help="dynamic GridNeRF equals --backbone dgrid --fp16")
parser.add_argument('-D2', action='store_true',
                    help="dynamic nerf equals --backbone hexplane2 --fp16")

parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--six_views', action='store_true',
                    help="six_views mode: save the images of the six views")
parser.add_argument('--eval_interval', type=int, default=1,
                    help="evaluate on the valid set every interval epochs")
parser.add_argument('--save_interval', type=int, default=10,
                    help="save model every interval epochs")
parser.add_argument('--max_keep_ckpt', type=int, default=1,
                    help="how many ckpts are kept")
                    
parser.add_argument('--test_interval', type=int, default=50,
                    help="test on the test set every interval epochs")
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--log_every', type=int, default=20,
                    help="log losses every X iterations")
parser.add_argument('--use_wandb', action='store_true',
                    help="log online into wandb")

# guidance
parser.add_argument('--guidance', type=str, nargs='*',
                    default=['SD'], help='guidance model')
parser.add_argument('--guidance_scale', type=float, nargs='*', default=[100],
                    help="diffusion model classifier-free guidance scale")
parser.add_argument('--gudiance_spatial_weighting',
                    action='store_true', help="add spatial weighting to guidance")
parser.add_argument('--save_train_every', type=int,
                    default=-1, help="save sds guidance")

# clip guidance
# lambda_clip, set to 1 if use clip loss outside sds
parser.add_argument('--lambda_clip', type=float, default=0,
                    help="loss scale for clip loss outside sds")
# set to 100 if use clip guidance in sds
parser.add_argument('--clip_version', type=str,
                    default='large', help="clip version, large is ued in stable diffusion")
parser.add_argument('--clip_guidance', type=float, default=0,
                    help="diffusion model classifier-free guidance scale")
parser.add_argument('--clip_t', type=float, default=0.4,
                    help="time step thresh started to use clip")
parser.add_argument('--clip_iterative', action='store_true',
                    help="use clipd iteratively with sds")
parser.add_argument('--clip_image_loss', action='store_true',
                    help="use image as reference in clip")
parser.add_argument('--save_guidance_every', type=int,
                    default=199, help="save sds guidance, since the 200 will do rgb loss, no guidance")

# 3D prior: Shap-E. Does not work.
parser.add_argument('--use_shape', action='store_true',
                    help="enable shap-e initization")
parser.add_argument('--shape_guidance', type=float, default=3,
                    help="guidance scaling for shap-e prior")
parser.add_argument('--shape_radius', type=float, default=4,
                    help="camera raidus for shap-e prior")
parser.add_argument('--shape_fovy', type=float, default=40,
                    help="fov for shap-e prior")
parser.add_argument('--shape_no_color', action='store_false',
                    dest='shape_init_color', help="do not use shap-E color for initization")
parser.add_argument('--shape_rpst', type=str, default='sdf',
                    help="use sdf to init NeRF/mesh by default")

# image options.
parser.add_argument('--image', default=None, help="image prompt")
parser.add_argument('--image_config', default=None, help="image config csv")
parser.add_argument('--learned_embeds_path', type=str,
                    default=None, help="path to learned embeds of the given image")
parser.add_argument('--known_iters', type=int, default=100,
                    help="loss scale for alpha entropy")
parser.add_argument('--known_view_interval', type=int, default=4,
                    help="do reconstruction every X iterations to save on compute")
parser.add_argument('--bg_color_known', type=str,
                    default=None, help='pixelnoise, noise, None')   # pixelnoise
parser.add_argument('--known_shading', type=str, default='lambertian')

# DMTet and Mesh options
parser.add_argument('--save_mesh', action='store_true',
                    help="export an obj mesh with texture")
parser.add_argument('--mcubes_resolution', type=int, default=256,
                    help="mcubes resolution for extracting mesh")
parser.add_argument('--decimate_target', type=int, default=5e4,
                    help="target face number for mesh decimation")
parser.add_argument('--dmtet', action='store_true',
                    help="use dmtet finetuning")
parser.add_argument('--tet_mlp', action='store_true',
                    help="use tet_mlp finetuning")
parser.add_argument('--base_mesh', default=None,
                    help="base mesh for dmtet init")
parser.add_argument('--tet_grid_size', type=int,
                    default=256, help="tet grid size")
parser.add_argument('--init_ckpt', type=str, default='',
                    help="ckpt to init dmtet")
parser.add_argument('--lock_geo', action='store_true',
                    help="disable dmtet to learn geometry")

# training options
parser.add_argument('--iters', type=int, default=5000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
parser.add_argument('--lr_scale_nerf', type=float,
                    default=1, help="max learning rate")
parser.add_argument('--lr_scale_time', type=float,
                    default=1, help="max learning rate")
parser.add_argument('--lr_time_iter', type=int,
                    default=0, help="the iteration starting to optimize the time layer")

parser.add_argument('--lr_scale_texture', type=float,
                    default=1, help="max learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--cuda_ray', action='store_true',
                    help="use CUDA raymarching instead of pytorch")
parser.add_argument('--taichi_ray', action='store_true',
                    help="use taichi raymarching")
parser.add_argument('--max_steps', type=int, default=1024,
                    help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=64,
                    help="num steps sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=32,
                    help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16,
                    help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096,
                    help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
parser.add_argument('--latent_iter_ratio', type=float, default=0.0,
                    help="training iters that only use latent normal shading")
parser.add_argument('--normal_iter_ratio', type=float, default=0.0,
                    help="training iters that only use normal shading")
parser.add_argument('--textureless_iter_ratio', type=float, default=0.0,
                    help="training iters that only use textureless shading")
parser.add_argument('--albedo_iter_ratio', type=float, default=0,
                    help="training iters that only use albedo shading")
parser.add_argument('--warmup_bg_color', type=str, default=None,
                    help="bg color [None | pixelnoise | noise | white]")
parser.add_argument('--bg_color', type=str, default=None)
parser.add_argument('--bg_color_test', default='white')
parser.add_argument('--ema_decay', type=float, default=0, #default=0.95,
                    help="exponential moving average of model weights")
parser.add_argument('--jitter_pose', action='store_true',
                    help="add jitters to the randomly sampled camera poses")
parser.add_argument('--jitter_center', type=float, default=0.2,
                    help="amount of jitter to add to sampled camera pose's center (camera location)")
parser.add_argument('--jitter_target', type=float, default=0.2,
                    help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
parser.add_argument('--jitter_up', type=float, default=0.02,
                    help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
parser.add_argument('--uniform_sphere_rate', type=float, default=0.5,
                    help="likelihood of sampling camera location uniformly on the sphere surface area")
parser.add_argument('--grad_clip', type=float, default=-1,
                    help="clip grad of all grad to this limit, negative value disables it")
parser.add_argument('--grad_clip_rgb', type=float, default=-1,
                    help="clip grad of rgb space grad to this limit, negative value disables it")
parser.add_argument('--grid_levels_mask', type=int, default=8,
                    help="the number of levels in the feature grid to mask (to disable use 0)")
parser.add_argument('--grid_levels_mask_iters', type=int, default=3000,
                    help="the number of iterations for feature grid masking (to disable use 0)")

# model options
parser.add_argument('--bg_radius', type=float, default=1.4,
                    help="if positive, use a background model at sphere(bg_radius)")
parser.add_argument('--density_activation', type=str, default='exp',
                    choices=['softplus', 'exp', 'relu'], help="density activation function")
parser.add_argument('--density_thresh', type=float, default=10,
                    help="threshold for density grid to be occupied")
# add more strength to the center, believe the center is more likely to have objects.
parser.add_argument('--blob_density', type=float, default=10,
                    help="max (center) density for the density blob")
parser.add_argument('--blob_radius', type=float, default=0.2,
                    help="control the radius for the density blob")
# network backbone
parser.add_argument('--backbone', type=str, default='grid',
                    choices=['grid', 'vanilla', 'grid4d'], help="nerf backbone")
parser.add_argument('--dynamic', action='store_true',
                    help="add dynamic layers to grid model")

parser.add_argument('--grid_type', type=str,
                    default='tiledgrid', help="grid type: hashgrid | tiledgrid | tiledgrid_triplane |hashgrid_triplane")
parser.add_argument('--hidden_dim_bg', type=int, default=32,
                    help="channels for background network")
parser.add_argument('--optim', type=str, default='adam',
                    choices=['adan', 'adam'], help="optimizer")
parser.add_argument('--sd_version', type=str, default='1.5',
                    help="stable diffusion version")
parser.add_argument('--hf_key', type=str, default=None,
                    help="hugging face Stable diffusion model key")
# try this if CUDA OOM
parser.add_argument('--fp16', action='store_true',
                    help="use float16 for training")
parser.add_argument('--vram_O', action='store_true',
                    help="optimization for low VRAM usage")
# rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
parser.add_argument('--w', type=int, default=128,
                    help="render width for NeRF in training")
parser.add_argument('--h', type=int, default=128,
                    help="render height for NeRF in training") 
parser.add_argument('--known_view_scale', type=float, default=1.5,
                    help="multiply --h/w by this for known view rendering")
parser.add_argument('--known_view_noise_scale', type=float, default=1e-3,
                    help="random camera noise added to rays_o and rays_d")
parser.add_argument('--noise_known_camera_annealing', action='store_true',
                    help="anneal the noise to zero over the coarse of training")
parser.add_argument('--dmtet_reso_scale', type=float, default=8,
                    help="multiply --h/w by this for dmtet finetuning")
parser.add_argument('--rm_edge', action='store_true',
                    help="remove edge (ideally only enale for high resolution cases)")
parser.add_argument('--edge_threshold', type=float, default=0.1,
                    help="remove edges with value > threshold")
parser.add_argument('--edge_width', type=float, default=5,
                    help="edge width")
parser.add_argument('--batch_size', type=int, default=1,
                    help="images to render per batch using NeRF")

# dataset options
parser.add_argument('--bound', type=float, default=1.0,
                    help="assume the scene is bounded in box(-bound, bound)")
parser.add_argument('--dt_gamma', type=float, default=0,
                    help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.1,
                    help="minimum near distance for camera")

parser.add_argument('--radius_range', type=float, nargs='*',
                    default=[1.8, 1.8], help="training camera radius range")
parser.add_argument('--theta_range', type=float, nargs='*',
                    default=[45, 135], help="training camera elevation/polar range, 90 is front")
parser.add_argument('--phi_range', type=float, nargs='*',
                    default=[-180, 180], help="training camera azimuth range")
parser.add_argument('--fovy_range', type=float, nargs='*',
                    default=[40, 40], help="training camera fovy range")

parser.add_argument('--default_radius', type=float, default=1.8,
                    help="radius for the default view") 
parser.add_argument('--default_polar', type=float,
                    default=90, help="polar for the default view")
parser.add_argument('--default_azimuth', type=float,
                    default=0, help="azimuth for the default view")
parser.add_argument('--default_fovy', type=float, default=40,
                    help="fovy for the default view") 

parser.add_argument('--progressive_view', action='store_true',
                    help="progressively expand view sampling range from default to full")
parser.add_argument('--progressive_level', action='store_true',
                    help="progressively increase gridencoder's max_level")

parser.add_argument('--angle_overhead', type=float, default=30,
                    help="[0, angle_overhead] is the overhead region")
parser.add_argument('--angle_front', type=float, default=60,
                    help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
parser.add_argument('--t_range', type=float, nargs='*',
                    default=[0.2, 0.6], help="stable diffusion time steps range")
parser.add_argument('--zero123_t_range', type=float, nargs='*',
                    default=None, help="Zero123 timestep range, now it can use a different with sd")

# regularizations
parser.add_argument('--lambda_entropy', type=float, default=1e-3,
                    help="loss scale for alpha entropy, favors 0 or 1")
# Try increasing/decreasing lambda_opacity if your scene is stuffed with floaters/becoming empty.
parser.add_argument('--lambda_opacity', type=float, default=0.,
                    help="loss scale for alpha value, avoid uncessary filling")
# Try increasing/decreasing lambda_orient if you object is foggy/over-smoothed.
parser.add_argument('--lambda_orient', type=float,
                    default=1e-2, help="loss scale for orientation")
parser.add_argument('--lambda_tv', type=float, default=0,
                    help="loss scale for total variation of grad")
parser.add_argument('--lambda_wd', type=float, default=0,
                    help="loss scale for weight decay of grad")
parser.add_argument('--lambda_normal_smooth', type=float, default=0.5,
                    help="loss scale for first-order 2D normal image smoothness")
parser.add_argument('--lambda_normal_smooth2d', type=float, default=0.5,
                    help="loss scale for second-order 2D normal image smoothness")
parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0.0,
                    help="loss scale for second-order 2D normal image smoothness")
parser.add_argument('--lambda_guidance', type=float, nargs='*',
                    default=[1], help="loss scale for SDS")
parser.add_argument('--lambda_rgb', type=float,
                    default=5, help="loss scale for RGB")
parser.add_argument('--lambda_mask', type=float, default=0.5,
                    help="loss scale for mask (alpha)")
parser.add_argument('--lambda_depth', type=float, default=0.01,
                    help="loss scale for relative depth of the known view")
parser.add_argument('--lambda_normal', type=float,
                    default=0.0, help="loss scale for normals of the known view")
parser.add_argument('--lambda_depth_mse', type=float, default=0.0,
                    help="loss scale for depth of the known view")
parser.add_argument('--no_normalize_depth', action='store_false', dest='normalize_depth', help="normalize depth")

# for DMTet
parser.add_argument('--lambda_mesh_normal', type=float,
                    default=0.1, help="loss scale for mesh normal smoothness")
parser.add_argument('--lambda_mesh_lap', type=float,
                    default=0.1, help="loss scale for mesh laplacian")

# GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=400, help="GUI width")
parser.add_argument('--H', type=int, default=400, help="GUI height")
parser.add_argument('--radius', type=float, default=1.8,
                    help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=40, 
                    help="default GUI camera fovy")
parser.add_argument('--light_theta', type=float, default=60,
                    help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
parser.add_argument('--light_phi', type=float, default=0,
                    help="default GUI light direction in [0, 360), azimuth")
parser.add_argument('--max_spp', type=int, default=1,
                    help="GUI rendering max sample per pixel")
parser.add_argument('--zero123_config', type=str,
                    default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
parser.add_argument('--zero123_ckpt', type=str,
                    default='./pretrained/zero123/zero123-xl.ckpt', help="ckpt for zero123")
parser.add_argument('--zero123_grad_scale', type=str, default='angle',
                    help="whether to scale the gradients based on 'angle' or 'None'")

parser.add_argument('--dataset_size_train', type=int, default=100,
                    help="Length of train dataset i.e. # of iterations per epoch")
parser.add_argument('--dataset_size_valid', type=int, default=8,
                    help="# of frames to render in the turntable video in validation")
parser.add_argument('--dataset_size_test', type=int, default=10,
                    help="# of frames to render in the turntable video at test time")

## dynamic finetune
parser.add_argument('--dynamic_ft', action='store_true', help="dynamic finetune")
parser.add_argument('--static_ckpt', type=str, default='')
parser.add_argument('--time_size', type=int, default=1, help="cuda ray time grid size")
parser.add_argument('--num_frames', type=int, default=16, help="number of frames for training")
parser.add_argument('--num_test_frames', type=int, default=64, help="number of frames for test")

## lr scale for encoder and deform net
parser.add_argument('--lr_encoder_scale', type=float, default=10.0, help="encoder lr scale, 1 in default, for dynamic it's better to set it to a smaller value")
parser.add_argument('--lr_deform_scale', type=float, default=1.0, help="deform net lr scale, 1 in default")
parser.add_argument('--lr_sigma_scale', type=float, default=1.0, help="sigma net lr scale, 1 in default")

parser.add_argument('--new_sds', action='store_true', help="use the new version of sds in stable dreamfusion")
parser.add_argument('--mean_sds', action='store_true', help="use the new version of sds and mean it")
parser.add_argument('--mean_frame', action='store_true', help="video sds mean over frame")

parser.add_argument('--rgbd_scale', type=float, default=1.0, help="rgb loss scale")


parser.add_argument('--num_chunk', type=int, default=1, help="num of chunk in the forward function")


## grid 4d
parser.add_argument('--lambda_time_tv', type=float, default=0.1, help="loss scale for total variation of time grid, in hexplane, it is set to 1e-4, in grid4d it should be set to 0.1 to 1")

## text embedding all
parser.add_argument('--text_emb_all', action='store_true', help="using text emb all")

## dynamic camera
parser.add_argument('--dynamic_cam_rate', type=float, default=0.0, help="dynamic camera rate")
parser.add_argument('--scale_delta', action='store_true', help="using text emb all")
parser.add_argument('--no_view_text', action='store_true', help="not use view in the text")
parser.add_argument('--view_text', action='store_true', help="use view in the text, if set to True, will change no_view_text to false ")

## time data sample
parser.add_argument('--start_from_zero', action='store_true', help="in dynamic stage, always sample the first frame at time 0")
parser.add_argument('--zero_rate', type=float, default=0.1, help="the rate of sampling start from zero")
parser.add_argument('--precision', type=int, default=256, help="precision of time sampling, default is 64")
parser.add_argument('--zero_precision', type=int, default=256, help="precision of time sampling when start from zero, default is 64")
parser.add_argument('--end_rate', type=float, default=0.1, help="the rate of sampling end at one")


## backbone config
parser.add_argument('--num_levels', type=int, default=16, help="number of multi-resolution level, default is 16, makeavid3d is 7")
parser.add_argument('--num_layers', type=int, default=2, help="number of layers of sigma net, default is 2, makeavid3d is 5")
parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim of sigma net, default is 64, makeavid3d is 128")
parser.add_argument('--level_dim', type=int, default=2, help="dim of encoder, default 2")
parser.add_argument('--color_net', action='store_true', help="use color net instead of albedo")

## grid4d model
parser.add_argument('--time_grid_size', type=int, default=16, help="grid4d model, number of time grids")
parser.add_argument('--freeze_first', action='store_true', help="freeze the first encoder layer")

## textureless
parser.add_argument('--textureless_rate', type=float, default=0.2, help="rate for textureless rendering, default is 0.2")
parser.add_argument('--no_textureless_condition', action='store_true', help="not use textureless for condition image")

## controlnet img2img control
parser.add_argument('--cn_base', type=str, default=None, help="controlnet base model ckpt")
parser.add_argument('--cn_scale', type=float, default=1.0, help="controlnet feature scale")
parser.add_argument('--cn_text', default=None, help="text prompt for contorlnet sr")
parser.add_argument('--cn_cn_text', default='best quality, high resolution', help="text prompt for controlnet part of model")
parser.add_argument('--cn_learned_embeds_path', type=str, default=None, help="path to learned embeds of the given image")
parser.add_argument('--cn_key', type=str, default='checkpoints/controlnet-v11/control_v11f1e_sd15_tile', help="controlnet key")
parser.add_argument('--cn_frames', type=int, default=16, help="number of frames used in cn model")
parser.add_argument('--cn_frame_method', type=str, default='even', help="how to choose cn_frames. even or random")
parser.add_argument('--cn_t_range', type=float, nargs='*', default=[0.2, 0.8], help="controlnet stable diffusion time steps range")
parser.add_argument('--cn_size', type=int, default=1024, help="size for the vid2vid super resolution model")
parser.add_argument('--fix_dynamic', action='store_true', help="fix dynamic model in cn control")
parser.add_argument('--grad_dyn', action='store_true', help="dynamic weight for sds loss, using ddim denoise method")


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    opt = edict(vars(args))

    opt.no_view_text = False if opt.view_text else opt.no_view_text

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'

    if opt.dynamic_ft:
        opt.dynamic = True
        assert opt.static_ckpt

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    if 'CN' in opt.guidance:
        if opt.cn_text is None:
            opt.cn_text = opt.text
        if opt.cn_learned_embeds_path is None:
            opt.cn_learned_embeds_path = opt.learned_embeds_path
            
        opt.fix_dynamic = True

    # parameters for image-conditioned generation
    if opt.image is not None or opt.image_config is not None:
        if 'zero123' in opt.guidance:
            # fix fov as zero123 doesn't support changing fov
            opt.fovy_range = [opt.default_fovy, opt.default_fovy]
        else:
            opt.known_view_interval = 2
        
        opt.start_from_zero = True

        # if 'SD' in opt.guidance:
        #     # opt.t_range = [0.2, 0.6] ## if this is the problem??
        #     ## NOTE the t_range args is used in both sd and zero123 guidance
        #     opt.bg_radius = -1

        # latent warmup is not needed
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0

        if opt.image is not None:
            opt.images += [opt.image]
            opt.ref_radii += [opt.default_radius]
            opt.ref_polars += [opt.default_polar]
            opt.ref_azimuths += [opt.default_azimuth]
            opt.zero123_ws += [opt.default_zero123_w]

        if opt.image_config is not None:
            # for multiview (zero123)
            conf = pd.read_csv(opt.image_config, skipinitialspace=True)
            opt.images += list(conf.image)
            opt.ref_radii += list(conf.radius)
            opt.ref_polars += list(conf.polar)
            opt.ref_azimuths += list(conf.azimuth)
            opt.zero123_ws += list(conf.zero123_weight)
            if opt.image is None:
                opt.default_radius = opt.ref_radii[0]
                opt.default_polar = opt.ref_polars[0]
                opt.default_azimuth = opt.ref_azimuths[0]
                opt.default_zero123_w = opt.zero123_ws[0]

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # default parameters for finetuning
    if opt.dmtet:
        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)
        opt.known_view_scale = 1
        opt.grid_levels_mask = -1 # disable corse nerf (fine to keep, not necesary)
        opt.t_range = [0.02, 0.50]  # ref: magic3D

        if opt.images is not None:
            opt.lambda_normal = 0
            opt.lambda_depth = 0

        # assume finetuning
        opt.latent_iter_ratio = 0
        opt.textureless_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.normal_iter_ratio = 0
        opt.progressive_view = False
        opt.progressive_level = False

    # record full range for progressive view expansion
    if opt.progressive_view:
        # disable as they disturb progressive view
        opt.jitter_pose = False
        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    opt.use_clip = opt.clip_guidance > 0 or opt.lambda_clip > 0
    # Do not support Shap-E for NeRF yet.
    opt.use_shape = False if not opt.dmtet else opt.use_shape

    # workspace prepare
    setup_workspace(opt)
    dnnultis.setup_logging(opt.log_path)

    if opt.seed < 0:
        opt.seed = random.randint(0, 10000)
    seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid4d':
        from dnerf.network_4dgrid import NeRFNetwork
    else:
        raise NotImplementedError(
            f'--backbone {opt.backbone} is not implemented!')


    model = NeRFNetwork(opt).to(device)
    # ipdb.set_trace()
    if opt.dynamic_ft:
        static_checkpoint = torch.load(opt.static_ckpt, map_location='cpu')
        model.from_pretrained_3d(static_checkpoint)
        from dnerf.utils import DTrainer as Trainer
        from dnerf.provider import DNeRFDataset as NeRFDataset
    
    if opt.cn_base:
        logger.info(f"Loading Pre-trained Dynamic NeRF Model: {opt.cn_base}" )
        sr_base_checkpoint = torch.load(opt.cn_base, map_location='cpu')
        model.load_state_dict(sr_base_checkpoint['model'])
        
        
    if opt.init_ckpt != '':
        if not os.path.exists(opt.init_ckpt):
            logger.warning(f'ckpt {opt.init_ckpt} is not found')
        else:
            # load pretrained weights to init dmtet
            state_dict = torch.load(opt.init_ckpt, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            logger.info(f'init from {opt.init_ckpt}...')
            # if init ckpt is provided, we assume the color network is well learned and do not need base_mesh init
            opt.shape_init_color = False
            opt.base_mesh = None

    if opt.use_shape and opt.dmtet:
        # now only supports shape for dmtet init
        from guidance.shape_utils import get_shape_from_image

        opt.points = generate_grid_points(
            128, device=device) if not opt.dmtet else model.dmtet.verts
        opt.rpsts, opt.colors = get_shape_from_image(
            opt.image.replace('rgba', 'rgb'),
            opt.points,
            rpst_type=opt.shape_rpst,
            get_color=opt.shape_init_color,
            shape_guidance=opt.shape_guidance, device=device)
        scale = opt.default_radius / opt.shape_radius * \
            np.tan(np.deg2rad(opt.default_fovy / 2)) / \
            np.tan(np.deg2rad(opt.shape_fovy / 2))
        if opt.dmtet:
            model.dmtet.reset_tet_scale(scale)
        else:
            opt.points *= scale
        logger.info(f'Got sdf from Shap-E init...')

    logger.info(model)

    if opt.six_views:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device,
                          workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(
            opt, device=device, type='six_views', H=opt.H, W=opt.W, size=6).dataloader(batch_size=1)
        trainer.test(test_loader, write_video=False)

        if opt.save_mesh:
            trainer.save_mesh()

    elif opt.test:
        guidance = None  # no need to load guidance model at test
        trainer = Trainer(' '.join(sys.argv), os.path.basename(opt.workspace), opt, model, guidance,
                          device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(
                opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader()
            trainer.test(test_loader)
            # trainer.test(test_loader, shading='normal') # save normal
            if opt.save_mesh:
                try:
                    trainer.save_mesh()
                except:
                    pass
    else:
        train_loader = NeRFDataset(
            opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR

            def optimizer(model): return Adan(model.get_params(
                5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            def optimizer(model): 
                return torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 1)  # fixed

        guidance = nn.ModuleDict()
        lambda_guidance, guidance_scale = {}, {}
        for idx, guidance_type in enumerate(opt.guidance):
            lambda_guidance[guidance_type] = opt.lambda_guidance[idx] if idx < len(
                opt.lambda_guidance) else opt.lambda_guidance[-1]
            guidance_scale[guidance_type] = opt.guidance_scale[idx] if idx < len(
                opt.guidance_scale) else opt.guidance_scale[-1]
            if 'SD' == guidance_type:
                from guidance.sd_utils import StableDiffusion, token_replace

                guidance['SD'] = StableDiffusion(opt, device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range,
                                                 learned_embeds_path=opt.learned_embeds_path,
                                                 use_clip=opt.use_clip, clip_t=opt.clip_t, clip_iterative=opt.clip_iterative, clip_version=opt.clip_version,
                                                 )
                if opt.learned_embeds_path is not None and os.path.exists(opt.learned_embeds_path):  # add textual inversion tokens to model
                    opt.text, opt.negative = token_replace(
                        opt.text, opt.negative, opt.learned_embeds_path)
                    logger.info(
                        f'prompt: {opt.text}, negative: {opt.negative}')
                    if opt.check_prompt:
                        guidance['SD'].check_prompt(opt)
                else:
                    if opt.image is not None:
                        opt.text = opt.text.replace('<token>', os.path.basename(os.path.dirname(opt.image)))
                    logger.warning('No learned_embeds_path provided, using the folowing pure text prompt with degraded performance: ' + opt.text)

            if 'CN' == guidance_type:
                from guidance.cn_utils import ControlNetStableDiffusion, token_replace
                guidance['CN'] = ControlNetStableDiffusion(opt, device, opt.fp16, opt.vram_O, None, None, opt.cn_t_range,
                                                 learned_embeds_path=opt.cn_learned_embeds_path,
                                                 use_clip=opt.use_clip, clip_t=opt.clip_t, clip_iterative=opt.clip_iterative, clip_version=opt.clip_version,
                                                 )
                if opt.cn_learned_embeds_path is not None and os.path.exists(opt.cn_learned_embeds_path):  # add textual inversion tokens to model
                    opt.cn_text, opt.negative = token_replace(
                        opt.cn_text, opt.negative, opt.cn_learned_embeds_path)
                    logger.info(
                        f'Using Textual Inversion for ConotrolNet \n prompt: {opt.text}, negative: {opt.negative}')
                    if opt.check_prompt:
                        guidance['CN'].check_prompt(opt)
                else:
                    if opt.image is not None:
                        opt.cn_text = opt.cn_text.replace('<token>', os.path.basename(os.path.dirname(opt.image)))
                    logger.warning('No learned_embeds_path provided, using the folowing pure text prompt for ControlNet: ' + opt.cn_text)

            if 'IF' == guidance_type:
                from guidance.if_utils import IF
                guidance['IF'] = IF(device, opt.vram_O, opt.t_range)

            if 'zero123' == guidance_type:
                # from guidance.zero123_utils import Zero123
                # guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config,
                                            #   ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
                from guidance.zero123_diffusers_utils import Zero123
                if opt.zero123_t_range is None:
                    opt.zero123_t_range = opt.t_range

                guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, vram_O=opt.vram_O, t_range=opt.zero123_t_range, opt=opt)

            if 'clip' == guidance_type:
                from guidance.clip_utils import CLIP
                guidance['clip'] = CLIP(device)
        opt.lambda_guidance = lambda_guidance
        opt.guidance_scale = guidance_scale

        logger.info(opt)
        trainer = Trainer(' '.join(sys.argv), os.path.basename(opt.workspace), opt, model,
                          guidance,
                          device=device, workspace=opt.workspace, optimizer=optimizer,
                          ema_decay=opt.ema_decay, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True,)
        trainer.default_view_data = train_loader._data.get_default_view_data()
        if opt.dynamic_ft:
            if trainer.global_step == 0: # train from static
                trainer.global_step = static_checkpoint['global_step']
                trainer.epoch = static_checkpoint['epoch']

        if opt.cn_base:
            if trainer.global_step < sr_base_checkpoint['global_step']:
                trainer.global_step = sr_base_checkpoint['global_step']
                trainer.epoch = sr_base_checkpoint['epoch']


        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(
                opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader()
            test_loader = NeRFDataset(
                opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader()
            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            # ipdb.set_trace()
            trainer.train(train_loader, valid_loader, test_loader, max_epoch)

            if opt.save_mesh:
                try:
                    trainer.save_mesh()
                except:
                    pass