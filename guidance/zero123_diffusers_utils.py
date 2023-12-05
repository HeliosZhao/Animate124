import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available, is_torch_version

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from tqdm import tqdm
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from zero123_pipeline.pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from zero123_pipeline.cc_projector import ProjectionModel
import os

import ipdb


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class Zero123(nn.Module):
    def __init__(self, device, fp16, zero123_version='zero123-xl',vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt
        self.new_sds = opt.get('new_sds', False)

        self.precision_t = torch.float16 if fp16 else torch.float32 #
        pretrained_model_path = f'checkpoints/{zero123_version}'
        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler" )

        # image encoding components
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder", revision="v2.0").to(dtype=self.precision_t)
        ## projector
        cc_projection = ProjectionModel.from_pretrained(pretrained_model_path, subfolder="cc_projection").to(dtype=self.precision_t)

        pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
            pretrained_model_path, 
            torch_dtype=self.precision_t, 
            image_encoder=image_encoder,
            cc_projection=cc_projection,
            safety_checker=None,
            requires_safety_checker=False,
            scheduler=self.scheduler,)

        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
            print("==> using xformers for less gpu memory cost")

        pipe.to(device)

        # if is_torch_version(">=", "2.0.0"):
        #     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.unet = pipe.unet 
        self.vae = pipe.vae
        self.cc_projection = pipe.cc_projection
        self.pipe = pipe

        ## reduce memory
        # pipe.enable_vae_slicing()
        # pipe.enable_vae_tiling()
        # pipe.enable_attention_slicing()

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.scheduler.num_train_timesteps

        

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience


    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [B, 3, 256, 256] in [0, 1]
        # ipdb.set_trace()
        ## CLIP embedding takes range 0-255
        f_cross = self.pipe._encode_image((x*255).int(), self.device, 1, False)

        ## VAE takes range -1-1
        x = x * 2 - 1
        f_concate = self.pipe.prepare_img_latents(image=x, batch_size=x.size(0), dtype=self.precision_t, device=self.device, do_classifier_free_guidance=False)

        return f_cross, f_concate

    def angle_between(self, sph_v1, sph_v2):
        def sph2cart(sv):
            r, theta, phi = sv[0], sv[1], sv[2]
            return torch.tensor([r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi), r * torch.cos(theta)])
        def unit_vector(v):
            return v / torch.linalg.norm(v)
        def angle_between_2_sph(sv1, sv2):
            v1, v2 = sph2cart(sv1), sph2cart(sv2)
            v1_u, v2_u = unit_vector(v1), unit_vector(v2)
            return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
        angles = torch.empty(len(sph_v1), len(sph_v2))
        for i, sv1 in enumerate(sph_v1):
            for j, sv2 in enumerate(sph_v2):
                angles[i][j] = angle_between_2_sph(sv1, sv2)
        return angles

    def train_step(self, embeddings, pred_rgb, polar, azimuth, radius, guidance_scale=3, as_latent=False, grad_scale=1, save_guidance_path:Path=None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        # adjust SDS scale based on how far the novel view is from the known view
        ref_radii = embeddings['ref_radii']
        ref_polars = embeddings['ref_polars']
        ref_azimuths = embeddings['ref_azimuths']
        v1 = torch.stack([radius + ref_radii[0], torch.deg2rad(polar + ref_polars[0]), torch.deg2rad(azimuth + ref_azimuths[0])], dim=-1)   # polar,azimuth,radius are all actually delta wrt default
        v2 = torch.stack([torch.tensor(ref_radii), torch.deg2rad(torch.tensor(ref_polars)), torch.deg2rad(torch.tensor(ref_azimuths))], dim=-1)
        angles = torch.rad2deg(self.angle_between(v1, v2)).to(self.device)
        if self.opt.zero123_grad_scale == 'angle':
            grad_scale = (angles.min(dim=1)[0] / (180/len(ref_azimuths))) * grad_scale  # rethink 180/len(ref_azimuths) # claforte: try inverting grad_scale or just fixing it to 1.0
        elif self.opt.zero123_grad_scale == 'None':
            grad_scale = 1.0 # claforte: I think this might converge faster...?
        else:
            assert False, f'Unrecognized `zero123_grad_scale`: {self.opt.zero123_grad_scale}'
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256)

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # Set weights acc to closeness in angle
        if len(ref_azimuths) > 1:
            inv_angles = 1/angles
            inv_angles[inv_angles > 100] = 100
            inv_angles /= inv_angles.max(dim=-1, keepdim=True)[0]
            inv_angles[inv_angles < 0.1] = 0
        else:
            inv_angles = torch.tensor([1.]).to(self.device)

        # Multiply closeness-weight by user-given weights
        zero123_ws = torch.tensor(embeddings['zero123_ws'])[None, :].to(self.device) * inv_angles
        zero123_ws /= zero123_ws.max(dim=-1, keepdim=True)[0]
        zero123_ws[zero123_ws < 0.1] = 0

        with torch.no_grad():
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            # x_in = torch.cat([latents_noisy] * 2)
            # t_in = torch.cat([t] * 2)

            noise_preds = []
            # Loop through each ref image
            for (zero123_w, c_crossattn, c_concat, ref_polar, ref_azimuth, ref_radius) in zip(zero123_ws.T,
                                                                                              embeddings['c_crossattn'], embeddings['c_concat'],
                                                                                              ref_polars, ref_azimuths, ref_radii):
                # polar,azimuth,radius are all actually delta wrt default
                p = polar + ref_polars[0] - ref_polar
                a = azimuth + ref_azimuths[0] - ref_azimuth
                a[a > 180] -= 360 # range in [-180, 180]
                r = radius + ref_radii[0] - ref_radius
                # T = torch.tensor([math.radians(p), math.sin(math.radians(-a)), math.cos(math.radians(a)), r])
                # T = T[None, None, :].to(self.device)
                T = torch.stack([torch.deg2rad(p), torch.sin(torch.deg2rad(-a)), torch.cos(torch.deg2rad(a)), r], dim=-1)[:, None, :]
                cond = {}
                # ipdb.set_trace()
                # len(T) = 1
                clip_emb = self.cc_projection(torch.cat([ c_crossattn.repeat(len(T), 1, 1), T], dim=-1))
                cond['c_crossattn'] = torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0).to(device=self.unet.device, dtype=self.unet.dtype)
                cond['c_concat'] = torch.cat([torch.zeros_like(c_concat).repeat(len(T), 1, 1, 1).to(self.device), c_concat.repeat(len(T), 1, 1, 1)], dim=0).to(device=self.unet.device, dtype=self.unet.dtype)

                latent_model_input = torch.cat([noisy_latents] * 2)
                latent_model_input = torch.cat([latent_model_input, cond['c_concat'] ], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond['c_crossattn'] ).sample

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_preds.append(zero123_w[:, None, None, None] * noise_pred)

        noise_pred = torch.stack(noise_preds).sum(dim=0) / zero123_ws.sum(dim=-1)[:, None, None, None]
        # ipdb.set_trace()
        w = (1 - self.alphas[t])
        ## self.alphas : larger t, smaller self.alphas -> larger w
        grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)
        # grad = w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)


        if save_guidance_path:
            os.makedirs(os.path.dirname(save_guidance_path), exist_ok=True)
            with torch.no_grad():
                if as_latent:
                    pred_rgb_256 = self.decode_latents(latents) # claforte: test!

                # visualize predicted denoised image
                pred_original_sample = self.decode_latents((noisy_latents - (1 - self.alphas[t]) ** (0.5) * noise_pred) / self.alphas[t] ** (0.5))
                result_hopefully_less_noisy_image = self.decode_latents(latents - w*(noise_pred - noise))
                # result_hopefully_less_noisy_image = self.decode_latents(self.scheduler.step(noisy_latents, t, noise_pred).prev_sample)

                # visualize noisier image
                result_noisier_image = self.decode_latents(noisy_latents)

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_256, pred_original_sample, result_noisier_image, result_hopefully_less_noisy_image],dim=-1)
                save_image(viz_images, save_guidance_path)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        # ipdb.set_trace()
        if self.new_sds:
            targets = (latents - grad).detach() #
            if self.opt.get('mean_sds', False):
                loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='mean') 
            else:
                loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0] # B,4,H,W -> sum over 4,H,W

        else:
            # grad = grad * grad_scale
            # grad = torch.nan_to_num(grad)
            latents.backward(gradient=grad, retain_graph=True)
            loss = grad.abs().mean().detach() # B,4,H,W all mean

        return loss

    # verification
    @torch.no_grad()
    def __call__(self,
            image, # image tensor [1, 3, H, W] in [0, 1]
            polar=0, azimuth=0, radius=0, # new view params
            scale=3, ddim_steps=50, ddim_eta=1, h=256, w=256, # diffusion params
            c_crossattn=None, # 1,1,768
            c_concat=None, # 1,4,32,32
            post_process=True,
            generator=None,
        ):

        if c_crossattn is None:
            c_crossattn, c_concat = self.get_img_embeds(image)

        T = torch.tensor([math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), radius])
        T = T[None, None, :].to(self.device) ## 1,1,4

        cond = {}
        # ipdb.set_trace()
        clip_emb = self.cc_projection(torch.cat([ c_crossattn, T], dim=-1))
        cond['c_crossattn'] = torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0).to(device=self.device, dtype=self.precision_t)
        cond['c_concat'] = torch.cat([torch.zeros_like(c_concat).to(self.device), c_concat], dim=0).to(device=self.device, dtype=self.precision_t)

        # produce latents loop
        if generator is not None:
            latents = torch.randn((image.size(0), 4, h // self.pipe.vae_scale_factor, w // self.pipe.vae_scale_factor), device=self.device, dtype=self.precision_t, generator=generator) * self.scheduler.init_noise_sigma
        else:
            latents = torch.randn((image.size(0), 4, h // self.pipe.vae_scale_factor, w // self.pipe.vae_scale_factor), device=self.device, dtype=self.precision_t) * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(ddim_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, cond['c_concat'] ], dim=1)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond['c_crossattn'] ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred.to(dtype=torch.float32), t, latents.to(dtype=torch.float32), eta=ddim_eta).prev_sample.to(cond['c_concat'].dtype)

        # ipdb.set_trace()
        imgs = self.decode_latents(latents).float()
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) # 0-1

        return image


    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = self.vae.encode(imgs).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents # [B, 4, 32, 32] Latent space image


if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('--fp16', action='store_true', help="use float16 for training") # no use now, can only run in fp32

    parser.add_argument('--polar', type=float, default=0, help='delta polar angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')
    parser.add_argument('--six_views', action='store_true')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    print(f'[INFO] loading model ...')
    zero123 = Zero123(device, opt.fp16, opt=opt)

    print(f'[INFO] running model ...')
    if opt.six_views:
        thetas = [90]*4 + [1e-6] + [180]
        phis = [0, 90, 180, -90, 0, 0]
    else:
        thetas = [opt.polar]
        phis = [opt.azimuth]

    # c_cross = torch.load('out/zero123-lightn/pyt_cross.pth').to(zero123.device)
    # c_concate = torch.load('out/zero123-lightn/pyt_concat.pth').to(zero123.device)

    save_dir = 'out/zero123-diffuser-eta0'
    os.makedirs(save_dir, exist_ok=True)

    for theta, phi in tqdm(zip(thetas, phis)):
        generator = torch.Generator(device=zero123.device)
        generator.manual_seed(0)

        outputs = zero123(image, polar=theta, azimuth=phi, radius=opt.radius, generator=generator, c_concat=None, c_crossattn=None, ddim_eta=0)
        plt.imshow(outputs[0])
        # plt.show()
        plt.savefig(save_dir +  f'/test-model-{theta}-{phi}.jpg')
