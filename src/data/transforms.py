"""Image/mask augmentation and normalization pipelines built on Albumentations.

Kept in one place so training, evaluation and the synthetic-data smoke tests all resize/
normalize identically. Mask channels are always resized with nearest-neighbour interpolation
so class ids are preserved exactly (no post-training class ordering surprises).
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int, augment: bool = True) -> A.Compose:
    ops: list[A.BasicTransform] = [A.Resize(image_size, image_size, interpolation=1)]
    if augment:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.15, contrast_limit=0.15),
            A.HueSaturationValue(p=0.2, hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10),
            A.GaussNoise(p=0.15),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.3),
        ]
    ops += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops, additional_targets={"mask2": "mask"})


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask2": "mask"},
    )


def denormalize(image: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization for a CHW float tensor/array -> HWC uint8 for display."""
    img = image.transpose(1, 2, 0) if image.shape[0] in (1, 3) else image
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)
