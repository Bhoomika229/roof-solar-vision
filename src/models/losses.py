"""Segmentation losses: multi-class soft Dice, and a combined Dice + Cross-Entropy loss.

Cross-entropy gives well-calibrated per-pixel gradients from the start of training; Dice
directly optimizes region overlap and is far less sensitive to the severe class imbalance
between background/roof pixels and small obstacle classes (chimneys, ladders, windows) that is
typical of RID-style superstructure masks. Combining both is standard practice in medical/
aerial segmentation literature for exactly this imbalance reason.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.clamp(min=0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid
            target_onehot = target_onehot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCrossEntropyLoss(nn.Module):
    """``loss = ce_weight * CrossEntropy + dice_weight * DiceLoss``."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
