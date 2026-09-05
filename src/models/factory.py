"""Common model construction interface so U-Net and SegFormer are interchangeable.

Both models expose ``forward(x) -> logits`` of shape ``(B, num_classes, H, W)`` at the input
resolution, which is all the training/evaluation loop relies on.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.segformer import SegFormerSegmentation
from src.models.unet import UNet
from src.utils.config import Config


def build_model(cfg: Config) -> nn.Module:
    model_cfg = cfg.model
    architecture = model_cfg.architecture.lower()
    num_classes = model_cfg.num_classes

    if architecture == "unet":
        return UNet(
            in_channels=model_cfg.get("in_channels", 3),
            num_classes=num_classes,
            base_channels=model_cfg.get("base_channels", 32),
            depth=model_cfg.get("depth", 4),
        )
    if architecture == "segformer":
        return SegFormerSegmentation(
            num_classes=num_classes,
            pretrained_name=model_cfg.get("pretrained_name", "nvidia/segformer-b0-finetuned-ade-512-512"),
            image_size=cfg.dataset.image_size,
        )
    raise ValueError(f"Unknown model architecture '{architecture}'. Choose 'unet' or 'segformer'.")
