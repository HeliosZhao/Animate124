import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from dpt import DPTDepthModel


class BLIP2:
    def __init__(self, device="cuda"):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, torch.float16
        )

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return generated_text


class DPT:
    def __init__(self, task="depth", device="cuda"):
        self.task = task
        self.device = device

        if task == "depth":
            path = hf_hub_download(
                repo_id="clay3d/omnidata", filename="omnidata_dpt_depth_v2.ckpt"
            )
            self.model = DPTDepthModel(backbone="vitb_rn50_384")
            self.aug = transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )

        else:  # normal
            path = hf_hub_download(
                repo_id="clay3d/omnidata", filename="omnidata_dpt_normal_v2.ckpt"
            )
            self.model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
            self.aug = transforms.Compose(
                [transforms.Resize((384, 384)), transforms.ToTensor()]
            )

        # load model
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == "depth":
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
            )
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(
                normal, size=(H, W), mode="bicubic", align_corners=False
            )
            normal = normal.cpu().numpy()
            return normal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument("--size", default=256, type=int, help="output resolution")
    parser.add_argument(
        "--border_ratio", default=0.2, type=float, help="output border ratio"
    )
    parser.add_argument(
        "--recenter",
        type=bool,
        default=True,
        help="recenter, potentially not helpful for multiview zero123",
    )
    parser.add_argument("--dont_recenter", dest="recenter", action="store_false")
    parser.add_argument("--need_caption", action="store_true")
    parser.add_argument("--use_existing_background", action="store_true")
    opt = parser.parse_args()

    out_dir = os.path.dirname(opt.path)
    out_rgba = os.path.join(
        out_dir, "_rgba.png"
    )
    out_depth = os.path.join(
        out_dir, "_depth.png"
    )
    out_normal = os.path.join(
        out_dir, "_normal.png"
    )
    out_caption = os.path.join(
        out_dir, "_caption.txt"
    )

    # load image
    print(f"[INFO] loading image...")
    image = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # carve background
    print(f"[INFO] background removal...")
    if not opt.use_existing_background:
        os.system("backgroundremover -i {} -o {}".format(opt.path, out_rgba))
    else:
        if os.path.exists(out_rgba):
            print(f"[INFO] using existing background: {out_rgba}")
        else:
            raise ValueError(f"cannot find existing background: {out_rgba}")
        opt.recenter = False
    carved_image = cv2.imread(out_rgba, cv2.IMREAD_UNCHANGED)
    carved_image = cv2.cvtColor(carved_image, cv2.COLOR_BGRA2RGBA)
    mask = carved_image[..., -1] > 0

    # predict depth
    print(f"[INFO] depth estimation...")
    dpt_depth_model = DPT(task="depth")
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (
        2 * depth[mask].mean() - depth[mask].min() + 1e-9
    )
    depth[~mask] = 0
    depth[depth > 1.0] = 1.0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model

    # predict normal
    print(f"[INFO] normal estimation...")
    dpt_normal_model = DPT(task="normal")
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model

    # recenter
    if opt.recenter:
        print(f"[INFO] recenter...")
        final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
        final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
        final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(opt.size * (1 - opt.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (opt.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (opt.size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            carved_image[x_min:x_max, y_min:y_max],
            (w2, h2),
            interpolation=cv2.INTER_AREA,
        )
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
        )
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
        )

    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal

    # write output
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)

    # predict caption (it's too slow... use your brain instead)
    if opt.need_caption:
        print(f"[INFO] captioning...")
        blip2 = BLIP2()
        caption = blip2(image)
        print(caption)
        with open(out_caption, "w") as f:
            f.write(caption)
