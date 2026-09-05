"""Roof-plane orientation (azimuth) estimation.

RID encodes each roof segment's azimuth directly as its semantic-segmentation class label
(N/NE/E/.../flat — see ``src/data/label_maps.py``), derived from true building-model geometry.
A model trained to predict the roof-segment class is therefore, by construction, predicting
orientation as well: no extra 3D reasoning step is needed or claimed here. This module turns a
predicted (or ground-truth) per-pixel segment mask into a per-segment and whole-roof orientation
summary, which is what a PV-siting workflow actually needs (azimuth drives insolation, which
drives yield).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.data.label_maps import COMPASS_BEARINGS, get_segment_classes


@dataclass
class OrientationResult:
    dominant_direction: str
    direction_pixel_fractions: dict[str, float]
    is_flat_dominant: bool
    mean_azimuth_deg: float | None
    source: str  # "model_prediction" | "ground_truth"


def _circular_mean_deg(directions: list[str], weights: list[float]) -> float | None:
    bearings = [COMPASS_BEARINGS[d] for d in directions if d in COMPASS_BEARINGS]
    if not bearings:
        return None
    ws = [w for d, w in zip(directions, weights) if d in COMPASS_BEARINGS]
    radians = np.radians(bearings)
    x = np.average(np.cos(radians), weights=ws)
    y = np.average(np.sin(radians), weights=ws)
    return float(np.degrees(np.arctan2(y, x)) % 360)


def estimate_orientation(
    segment_mask: np.ndarray,
    segment_scheme: int = 9,
    source: str = "model_prediction",
) -> OrientationResult:
    """Summarize roof orientation from a per-pixel segment-class mask.

    Class 0 is always "background" (not roof) and is excluded from the direction tally.
    """
    classes = get_segment_classes(segment_scheme)  # excludes background, includes "flat"
    class_ids_with_bg = ["background", *classes]

    ids, counts = np.unique(segment_mask, return_counts=True)
    tally: Counter = Counter()
    total_roof_pixels = 0
    for cid, count in zip(ids, counts):
        if cid == 0 or cid >= len(class_ids_with_bg):
            continue
        name = class_ids_with_bg[int(cid)]
        tally[name] += int(count)
        total_roof_pixels += int(count)

    if total_roof_pixels == 0:
        return OrientationResult("unknown", {}, True, None, source)

    fractions = {name: count / total_roof_pixels for name, count in tally.items()}
    dominant = max(tally, key=tally.get)
    is_flat = dominant == "flat"

    non_flat_dirs = [d for d in tally if d != "flat"]
    non_flat_weights = [tally[d] for d in non_flat_dirs]
    mean_az = _circular_mean_deg(non_flat_dirs, non_flat_weights) if non_flat_dirs else None

    return OrientationResult(
        dominant_direction=dominant,
        direction_pixel_fractions=fractions,
        is_flat_dominant=is_flat,
        mean_azimuth_deg=mean_az,
        source=source,
    )


def direction_to_compass_label(direction: str) -> str:
    labels = {
        "N": "North", "NNE": "North-Northeast", "NE": "Northeast", "ENE": "East-Northeast",
        "E": "East", "ESE": "East-Southeast", "SE": "Southeast", "SSE": "South-Southeast",
        "S": "South", "SSW": "South-Southwest", "SW": "Southwest", "WSW": "West-Southwest",
        "W": "West", "WNW": "West-Northwest", "NW": "Northwest", "NNW": "North-Northwest",
        "flat": "Flat (no dominant facing)",
    }
    return labels.get(direction, direction)
