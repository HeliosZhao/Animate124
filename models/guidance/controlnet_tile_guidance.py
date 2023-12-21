import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline, DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from .controlnet.stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

def center_crop_to_square(tensor):
    _, _, h, w = tensor.shape
    min_size = min(h, w)
    
    # Calculate the top, left corner of the cropped image
    top = (h - min_size) // 2
    left = (w - min_size) // 2

    # Crop the image
    cropped_tensor = tensor[:, :, top:top + min_size, left:left + min_size]

    return cropped_tensor


@threestudio.register("animate124-controlnet-tile-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        # ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        control_type: str = "tile"  # tile
        controlnet_name_or_path: str = ""
        controlnet_prompt: str = "best quality"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1
        fixed_frames: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path = self.cfg.controlnet_name_or_path

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        sd_pipe = DiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs)

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=sd_pipe.vae,
            text_encoder=sd_pipe.text_encoder,
            tokenizer=sd_pipe.tokenizer,
            unet=sd_pipe.unet,
            controlnet=controlnet,
            scheduler=sd_pipe.scheduler,
            safety_checker=sd_pipe.safety_checker,
            feature_extractor=sd_pipe.feature_extractor,
            requires_safety_checker=False
        ).to(self.device, self.weights_dtype)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        # self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        # if self.cfg.control_type == "normal":
        #     self.preprocessor = NormalBaeDetector.from_pretrained(
        #         "lllyasviel/Annotators"
        #     )
        #     self.preprocessor.model.to(self.device)
        # elif self.cfg.control_type == "canny":
        #     self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None
        self.get_controlnet_text_embeddings(self.cfg.controlnet_prompt)

        threestudio.info(f"Loaded ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_controlnet_text_embeddings(self,
        prompt
    ):
        tokens = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(tokens.input_ids.to(self.pipe.text_encoder.device))[0]
        self.controlnet_text_embeddings = text_embeddings.to(dtype=self.weights_dtype, device=self.device)
    

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond: Float[Tensor, "B 3 H W"],
        t: Int[Tensor, "B"],
        controlnet_text_embeddings: Float[Tensor, "BB 77 768"],
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=controlnet_text_embeddings,
                image_cond=torch.cat([image_cond] * 2) ,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape # batch size is number of frames
        # assert batch_size == 1
        assert rgb.shape[:-1] == cond_rgb.shape[:-1]
        ## randomly choose frames
        if self.cfg.fixed_frames > 0:
            indices = torch.randperm(rgb.size(0))[:self.cfg.fixed_frames]
            rgb, cond_rgb = rgb[indices], cond_rgb[indices]
            batch_size = self.cfg.fixed_frames

        ## add support to rectangle images, center crop to square
        rgb_BCHW = center_crop_to_square( rgb.permute(0, 3, 1, 2) ) # B,C,H,W
        cond_rgb_BCHW = center_crop_to_square( cond_rgb.permute(0, 3, 1, 2).detach() )
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = 1024, 1024
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        # image_cond = self.prepare_image_cond(cond_rgb)
        image_cond = F.interpolate(
            cond_rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )

        temp = torch.zeros(1).to(rgb.device)
        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        
        cond_text_embeddings, uncond_text_embeddings = text_embeddings.chunk(2, dim=0)
        text_embeddings = torch.cat([cond_text_embeddings.repeat(batch_size,1,1), uncond_text_embeddings.repeat(batch_size,1,1)], dim=0)
        controlnet_text_embeddings = torch.cat([self.controlnet_text_embeddings.repeat(batch_size,1,1), uncond_text_embeddings.repeat(batch_size,1,1).to(dtype=self.weights_dtype, device=self.device)], dim=0)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t, controlnet_text_embeddings)
        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }



    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
