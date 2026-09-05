"""Run a trained model over a dataloader and produce a full metrics report."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import SegmentationMetrics
from src.utils.device import get_device


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    class_names: list[str],
    mask_key: str = "segment_mask",
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or get_device()
    model.to(device)
    model.eval()

    metrics = SegmentationMetrics(num_classes)
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch[mask_key]
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        metrics.update(preds, targets)

    return metrics.compute_named(class_names)
