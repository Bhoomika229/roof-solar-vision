"""Dataset statistics used by the exploration notebook/script and sanity checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.dataset import RoofDataset


@dataclass
class DatasetStats:
    num_samples: int
    image_sizes: list[tuple[int, int]]
    segment_class_pixel_counts: dict[str, int] = field(default_factory=dict)
    superstructure_class_pixel_counts: dict[str, int] = field(default_factory=dict)
    roof_pixel_fraction_per_image: list[float] = field(default_factory=list)
    obstacle_pixel_fraction_per_image: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        sizes = np.array(self.image_sizes) if self.image_sizes else np.zeros((0, 2))
        return {
            "num_samples": self.num_samples,
            "mean_image_size": sizes.mean(axis=0).tolist() if len(sizes) else None,
            "unique_image_sizes": sorted({tuple(s) for s in self.image_sizes}),
            "segment_class_pixel_counts": self.segment_class_pixel_counts,
            "superstructure_class_pixel_counts": self.superstructure_class_pixel_counts,
            "mean_roof_pixel_fraction": float(np.mean(self.roof_pixel_fraction_per_image)) if self.roof_pixel_fraction_per_image else None,
            "mean_obstacle_pixel_fraction": float(np.mean(self.obstacle_pixel_fraction_per_image)) if self.obstacle_pixel_fraction_per_image else None,
        }


def compute_dataset_stats(dataset: RoofDataset, max_samples: int | None = None) -> DatasetStats:
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    image_sizes: list[tuple[int, int]] = []
    seg_counter: Counter = Counter()
    obs_counter: Counter = Counter()
    roof_fracs: list[float] = []
    obstacle_fracs: list[float] = []

    for i in range(n):
        item = dataset[i]
        seg_mask = item["segment_mask"].numpy() if hasattr(item["segment_mask"], "numpy") else np.asarray(item["segment_mask"])
        obs_mask = item["superstructure_mask"].numpy() if hasattr(item["superstructure_mask"], "numpy") else np.asarray(item["superstructure_mask"])
        image_sizes.append((seg_mask.shape[0], seg_mask.shape[1]))

        for cls_id, count in zip(*np.unique(seg_mask, return_counts=True)):
            name = dataset.segment_classes[int(cls_id)] if int(cls_id) < len(dataset.segment_classes) else f"id_{cls_id}"
            seg_counter[name] += int(count)
        for cls_id, count in zip(*np.unique(obs_mask, return_counts=True)):
            name = dataset.superstructure_classes[int(cls_id)] if int(cls_id) < len(dataset.superstructure_classes) else f"id_{cls_id}"
            obs_counter[name] += int(count)

        total = seg_mask.size
        roof_pixels = int((seg_mask != 0).sum())
        obstacle_pixels = int((obs_mask != 0).sum())
        roof_fracs.append(roof_pixels / total)
        obstacle_fracs.append(obstacle_pixels / total)

    return DatasetStats(
        num_samples=n,
        image_sizes=image_sizes,
        segment_class_pixel_counts=dict(seg_counter),
        superstructure_class_pixel_counts=dict(obs_counter),
        roof_pixel_fraction_per_image=roof_fracs,
        obstacle_pixel_fraction_per_image=obstacle_fracs,
    )
