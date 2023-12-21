import math
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from torch.utils.data import DataLoader, Dataset, IterableDataset


from .video_uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)

from threestudio.data.image import (
    SingleImageDataModuleConfig,
    SingleImageIterableDataset
)

class RandomImageVideoCameraIterableDataset(IterableDataset, Updateable):
    def __init__(
        self, cfg_single_view: Any, cfg_image: Any,
    ) -> None:
        super().__init__()
        self.cfg_single = parse_structured(
            RandomCameraDataModuleConfig, cfg_single_view
        )
        self.cfg_image = parse_structured(
            SingleImageDataModuleConfig, cfg_image
        )
        self.train_dataset_single = RandomCameraIterableDataset(self.cfg_single)
        self.train_dataset_image = SingleImageIterableDataset(self.cfg_image, 'train')
        self.idx = 0


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.train_dataset_single.update_step(epoch, global_step, on_load_weights)
        # self.train_dataset_image.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:

        batch = self.train_dataset_single.collate(batch)
        batch["single_view"] = True
        batch["ref_image"] = self.train_dataset_image.collate(None)
        ## set frame time as 0 
        batch["ref_image"]["frame_times"] = torch.FloatTensor([0.])
        batch["ref_image"]["random_camera"]["frame_times"] = torch.FloatTensor([0.])

        self.idx += 1

        return batch


@register("animate124-image-video-combined-camera-datamodule")
class ImageVideoCombinedCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_image = parse_structured(
            SingleImageDataModuleConfig, cfg.image
        )
        self.cfg_single = parse_structured(
            RandomCameraDataModuleConfig, cfg.single_view
        )

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomImageVideoCameraIterableDataset(
                self.cfg_single,
                self.cfg_image,
            )
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg_single, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg_single, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
