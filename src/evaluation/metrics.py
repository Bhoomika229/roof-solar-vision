"""Semantic segmentation metrics: per-class + aggregate IoU, Dice/F1, precision, recall, accuracy.

Implemented from a running confusion matrix (accumulated across batches) rather than per-batch
averages, which is the statistically correct way to report dataset-level IoU/Dice — per-batch
averaging biases small-object classes when batch composition is imbalanced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = preds.numpy().reshape(-1)
        targets_np = targets.numpy().reshape(-1)
        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np, targets_np = preds_np[valid], targets_np[valid]
        indices = targets_np * self.num_classes + preds_np
        binc = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.confusion[:] = 0

    def compute(self) -> dict[str, Any]:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        support = cm.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
            dice = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
            precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)

        pixel_accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else float("nan")
        present = support > 0

        return {
            "confusion_matrix": self.confusion.tolist(),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_support": support.astype(int).tolist(),
            "mean_iou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "mean_dice": float(np.nanmean(dice[present])) if present.any() else float("nan"),
            "mean_precision": float(np.nanmean(precision[present])) if present.any() else float("nan"),
            "mean_recall": float(np.nanmean(recall[present])) if present.any() else float("nan"),
            "pixel_accuracy": float(pixel_accuracy),
        }

    def compute_named(self, class_names: list[str]) -> dict[str, Any]:
        summary = self.compute()
        named = {"overall": {
            "mean_iou": summary["mean_iou"],
            "mean_dice": summary["mean_dice"],
            "mean_precision": summary["mean_precision"],
            "mean_recall": summary["mean_recall"],
            "pixel_accuracy": summary["pixel_accuracy"],
        }, "per_class": {}}
        for i, name in enumerate(class_names):
            named["per_class"][name] = {
                "iou": summary["per_class_iou"][i],
                "dice": summary["per_class_dice"][i],
                "precision": summary["per_class_precision"][i],
                "recall": summary["per_class_recall"][i],
                "support": summary["per_class_support"][i],
            }
        named["confusion_matrix"] = summary["confusion_matrix"]
        return named
