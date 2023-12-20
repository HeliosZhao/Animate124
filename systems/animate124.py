import os
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from .base import BaseLift3DSystem
import torch.nn.functional as F


@threestudio.register("animate124-system")
class Animate124(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"

        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)

        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)

        prompt_processor_type: str = ""
        prompt_processor: dict = field(default_factory=dict)

        ## guidance for ControlNet semantic refine
        guidance_cn_type: str = ""
        guidance_cn: dict = field(default_factory=dict)

        prompt_processor_cn_type: str = ""
        prompt_processor_cn: dict = field(default_factory=dict)
        sr: bool = False # semantic refine

        simultan: bool = False

        visualize_samples: bool = False
        prob_multi_view: Optional[float] = None
        prob_single_view_video: Optional[float] = None
        eval_depth_range_perc: Tuple[float, float] = (
            10,
            99,
        )  # Adjust manually based on object, near far depth bounds percentage in (0, 100)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.simultan = self.cfg.get("simultan", False)
        self.static = self.cfg.geometry.pos_encoding_config.get("static", True)
        self.sr = self.cfg.get("sr", False)

        self.guidance = None
        self.prompt_processor = None
        self.prompt_utils = None
        self.geometry_encoding = self.geometry.encoding.encoding

        ## prompt_processor_type_video is T2I if in static stage

        self.guidance_video = threestudio.find(self.cfg.guidance_type)(
            self.cfg.guidance
        )
        self.prompt_processor_video = threestudio.find(
            self.cfg.prompt_processor_type
        )(self.cfg.prompt_processor)
        self.prompt_utils_video = self.prompt_processor_video()
        if self.guidance is None:
            self.guidance = self.guidance_video
            self.prompt_processor = self.prompt_processor_video
            self.prompt_utils = self.prompt_utils_video

        self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
            self.cfg.guidance_3d
        )

        if self.sr:
            self.guidance_cn = threestudio.find(self.cfg.guidance_cn_type)(
                self.cfg.guidance_cn
            )
            prompt_processor_cn_class = threestudio.find(self.cfg.prompt_processor_cn_type)
            self.prompt_processor_cn = prompt_processor_cn_class(self.cfg.prompt_processor_cn)
            self.prompt_utils_cn = self.prompt_processor_cn()


    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        if not self.static:
            render_outs = []
            # TODO: Handle batch size higher than 1
            batch["frame_times"] = batch["frame_times"].flatten()
            for frame_idx, frame_time in enumerate(batch["frame_times"].tolist()):
                self.geometry_encoding.frame_time = frame_time
                if batch.get("train_dynamic_camera", False):
                    batch_frame = {}
                    for k_frame, v_frame in batch.items():
                        if isinstance(v_frame, torch.Tensor):
                            if v_frame.shape[0] == batch["frame_times"].shape[0]:
                                v_frame_up = v_frame[[frame_idx]].clone()
                            else:
                                v_frame_up = v_frame.clone()
                        else:
                            v_frame_up = v_frame
                        batch_frame[k_frame] = v_frame_up
                    render_out = self.renderer(**batch_frame)
                else:
                    render_out = self.renderer(**batch)
                render_outs.append(render_out)
            out = {}
            for k in render_out:
                out[k] = torch.cat(
                    [render_out_i[k] for render_out_i in render_outs]
                )
        else:
            out = self.renderer(**batch)

        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        '''
        batch["ref_image"] contains the reference frame information
        '''
        is_video = batch["is_video"]
        batch_size = batch["c2w"].shape[0]
        if batch["train_dynamic_camera"]:
            batch_size = batch_size // batch["frame_times"].shape[0]

        guidance = self.guidance_video
        prompt_utils = self.prompt_utils_video
        
        if is_video:
            static = self.static
            self.geometry_encoding.is_video = True
            self.geometry_encoding.set_temp_param_grad(True)
        else:
            static = True
            num_static_frames = 1  # Use a single random time for static guidance
            batch["frame_times"] = batch["frame_times"][
                torch.randperm(batch["frame_times"].shape[0])
            ][:num_static_frames]
            self.geometry_encoding.is_video = False
            self.geometry_encoding.set_temp_param_grad(False)

        out = self(batch)

        out_ref = None
        out_first = None
        if self.static:
            out_ref = self(batch["ref_image"])
        
        else:
            batch["num_frames"] = self.cfg.geometry.pos_encoding_config.num_frames
            if batch["start_from_zero"]:
                out_ref = self(batch["ref_image"])
                out_first = self(batch["ref_image"]["random_camera"])

        guidance_inp = out["comp_rgb"] # random camera
        guidance_out = guidance(
            guidance_inp, prompt_utils, **batch, rgb_as_latents=False
        )
        
        if static:
            guidance_3d_out = self.guidance_3d(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
            )

        elif batch['start_from_zero']:
            guidance_3d_out = self.guidance_3d(
                out_first["comp_rgb"],
                **batch["ref_image"]["random_camera"],
                rgb_as_latents=False,
            )
        else:
            guidance_3d_out = {}

        if self.sr:
            ## controlnet sds
            guidance_cn_out = self.guidance_cn(
                out["comp_rgb"], out["comp_rgb"].detach(), self.prompt_utils_cn, **batch, rgb_as_latents=False
            )
        else:
            guidance_cn_out = {}
            


        loss = 0.0

        for name, value in guidance_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])


        for name, value in guidance_3d_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}_3d", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_3d_")]
                )

        for name, value in guidance_cn_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}_cn", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_cn_")]
                )

        if out_ref is not None:
            batch_ref = batch["ref_image"]
            loss_rgb = F.mse_loss(
                out_ref["comp_rgb"],
                batch_ref["rgb"] * batch_ref["mask"].float()
                + out_ref["comp_rgb_bg"] * (1.0 - batch_ref["mask"].float()),
            )
            self.log("train/loss_rgb", loss_rgb)
            loss += loss_rgb * self.C(self.cfg.loss.lambda_rgb)

            with torch.cuda.amp.autocast(enabled=False):
                loss_mask = F.binary_cross_entropy(
                    out_ref["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                    batch_ref["mask"].float(),
                )
            self.log("train/loss_mask", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss.lambda_mask)


        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smoothness loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                loss_normal_smoothness_2d = (
                    normal[:, 1:, :, :] - normal[:, :-1, :, :]
                ).square().mean() + (
                    normal[:, :, 1:, :] - normal[:, :, :-1, :]
                ).square().mean()
                self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(
                    self.cfg.loss.lambda_normal_smoothness_2d
                )


            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

        if not self.static:
            batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_video["frame_times"] = batch["frame_times_video"]
            out_video = self(batch_video)
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_video.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out_video["comp_rgb"],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out_video
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out_video["comp_normal"],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out_video
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out_video["opacity"],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="validation_step",
                step=self.true_global_step,
                video=True,
            )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample(
                            self.prompt_utils_single_view,
                            **batch,
                            seed=self.global_step,
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample_lora(
                            self.prompt_utils_single_view, **batch
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.out_depths = []
        out = self(batch)
        depth = out["depth"][0, :, :, 0].detach().cpu().numpy()
        self.out_depths.append(depth)
        if "comp_rgb" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if "comp_normal" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if not self.static:
            batch_static = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_static["frame_times"] = torch.zeros_like(batch["frame_times"])
            out = self(batch_static)
            if "comp_rgb" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_rgb"][0],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )
            if "comp_normal" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test-normal_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )

    def on_test_epoch_end(self):
        fps = 15
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=fps,
            name=f"test",
            step=self.true_global_step,
        )
        if not self.static:
            self.save_img_sequence(
                f"it{self.true_global_step}-test_static",
                f"it{self.true_global_step}-test_static",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test_static",
                step=self.true_global_step,
            )
        out_depths = np.stack(self.out_depths)
        non_zeros_depth = out_depths[out_depths != 0]
        self.visu_perc_min_depth = np.percentile(
            non_zeros_depth, self.cfg.eval_depth_range_perc[0]
        )
        self.visu_perc_max_depth = np.percentile(
            non_zeros_depth, self.cfg.eval_depth_range_perc[1]
        )
        depth_color_maps = ["jet"]
        for depth_color_map in depth_color_maps:
            for i, depth in enumerate(out_depths):
                self.save_image_grid(
                    f"it{self.true_global_step}-test-depth-{depth_color_map}/{i}.png",
                    [
                        {
                            "type": "grayscale",
                            "img": depth,
                            "kwargs": {
                                "cmap": depth_color_map,
                                "data_range": "nonzero",
                            },
                        },
                    ],
                    name="depth_test_step",
                    step=self.true_global_step,
                )
        extra_renderings = [
            f"depth-{depth_color_map}" for depth_color_map in depth_color_maps
        ]
        for extra_rendering in extra_renderings:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-{extra_rendering}",
                f"it{self.true_global_step}-test-{extra_rendering}",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test",
                step=self.true_global_step,
            )
