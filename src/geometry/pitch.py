"""Approximate roof-pitch (slope) estimation from a single nadir image — a documented heuristic.

IMPORTANT — scientific honesty statement (see also CLAUDE.md and docs/research_report.md):
A single top-down (nadir) aerial photograph contains **no direct geometric cue for out-of-plane
tilt** without an external reference (stereo pair, known object height, shadow + sun-angle
metadata, or a digital surface model). RID itself sources its ground-truth ``slope`` values from
external LoD2 cadastral building models, not from the imagery. Any pipeline that claims to
"measure" pitch from a single RGB image alone is overclaiming.

What this module actually does is a **defensible approximation**: it classifies the roof into a
coarse archetype from the segmentation mask (flat / single-pitch / gable / hip-or-complex) using
only cues that *are* visible in a nadir image — the number and geometric arrangement of distinct
roof-segment orientation classes — and then reports a **regional architectural prior** for pitch
conditioned on that archetype (e.g. Central-European residential gable roofs typically fall in a
25-45 degree range), together with an explicitly wide uncertainty band. This is presented to the
user as an *estimate*, never as a measurement, and a formal citation for the prior is included in
``docs/research_report.md``.

If a true elevation source (stereo photogrammetry, LiDAR-derived nDSM) is later plugged in, it
should replace this heuristic entirely rather than be blended into it silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.label_maps import get_segment_classes

# Regional prior (documented assumption, not a per-building measurement): typical pitch ranges
# for Central-European residential roofs by coarse archetype, informally consistent with the
# slope values (~35 degrees) observed in RID's own sample vector labels for gabled roofs.
ARCHETYPE_PITCH_PRIOR_DEG: dict[str, tuple[float, float, float]] = {
    # archetype: (low, typical, high)
    "flat": (0.0, 2.0, 5.0),
    "single_pitch": (10.0, 20.0, 30.0),
    "gable_or_hip": (25.0, 35.0, 45.0),
    "complex": (20.0, 32.0, 48.0),
}


@dataclass
class PitchEstimate:
    archetype: str
    pitch_deg_low: float
    pitch_deg_typical: float
    pitch_deg_high: float
    method: str = "archetype_heuristic"
    is_measurement: bool = False


def classify_roof_archetype(segment_mask: np.ndarray, segment_scheme: int = 9, min_fraction: float = 0.05) -> str:
    classes = get_segment_classes(segment_scheme)
    class_ids_with_bg = ["background", *classes]
    ids, counts = np.unique(segment_mask, return_counts=True)

    tally: dict[str, int] = {}
    total = 0
    for cid, count in zip(ids, counts):
        if cid == 0 or cid >= len(class_ids_with_bg):
            continue
        name = class_ids_with_bg[int(cid)]
        tally[name] = tally.get(name, 0) + int(count)
        total += int(count)

    if total == 0:
        return "flat"

    significant = {name: c for name, c in tally.items() if c / total >= min_fraction}
    non_flat = {k: v for k, v in significant.items() if k != "flat"}

    if "flat" in significant and not non_flat:
        return "flat"
    if len(non_flat) <= 1:
        return "single_pitch"
    if len(non_flat) == 2:
        return "gable_or_hip"
    return "complex"


def estimate_pitch(segment_mask: np.ndarray, segment_scheme: int = 9) -> PitchEstimate:
    archetype = classify_roof_archetype(segment_mask, segment_scheme)
    low, typical, high = ARCHETYPE_PITCH_PRIOR_DEG[archetype]
    return PitchEstimate(archetype=archetype, pitch_deg_low=low, pitch_deg_typical=typical, pitch_deg_high=high)
