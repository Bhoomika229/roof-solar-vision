"""Canonical label taxonomy for the Roof Information Dataset (RID) and this project.

The class names below are **not guessed**. They are taken verbatim from the official RID
repository (TUMFTM/RID, https://github.com/TUMFTM/RID), specifically ``definitions.py``:

    label_classes_superstructures_annotation_experiment = [
        'pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow', 'tree', 'unknown'
    ]
    label_classes_segments_10 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'flat']

RID's own dictionaries are 0-indexed over exactly those lists, i.e. they reserve no explicit
background index (index 0 there is already a foreground class). For semantic-segmentation
masks in *this* project we prepend an explicit ``background`` class at index 0 so that a
standard pixel-wise cross-entropy/Dice setup has a well-defined "not this class" label. If you
plug in the official RID raster masks directly, use ``RID_RAW_SUPERSTRUCTURE_CLASSES`` /
``RID_RAW_SEGMENT_CLASSES_9`` (no background) and remap via ``remap_raw_to_canonical`` in
``src/data/dataset.py`` — the exact raw pixel-value encoding depends on how RID rasterized its
vector labels and should be verified against the downloaded mask files before training.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Roof superstructures (obstacles) — RID annotation-experiment class set.
# ---------------------------------------------------------------------------
RID_RAW_SUPERSTRUCTURE_CLASSES: list[str] = [
    "pvmodule", "dormer", "window", "ladder", "chimney", "shadow", "tree", "unknown",
]

SUPERSTRUCTURE_CLASSES: list[str] = ["background", *RID_RAW_SUPERSTRUCTURE_CLASSES]

# Which superstructure classes actually occlude/obstruct usable PV area (vs. purely
# informational classes like an already-present PV module, which is "usable but occupied",
# or "shadow", which is a soft rather than a hard obstacle). Documented, configurable choice.
HARD_OBSTACLE_CLASSES: set[str] = {"dormer", "window", "ladder", "chimney", "tree", "unknown"}
SOFT_OBSTACLE_CLASSES: set[str] = {"shadow"}
EXISTING_PV_CLASSES: set[str] = {"pvmodule"}

# ---------------------------------------------------------------------------
# Roof segments — orientation classes. RID offers three granularities; we default to the
# 9-class (8 compass directions + flat) scheme as the best trade-off between geometric
# usefulness for azimuth-dependent PV yield and having enough training examples per class.
# ---------------------------------------------------------------------------
RID_RAW_SEGMENT_CLASSES_5: list[str] = ["N", "E", "S", "W", "flat"]
RID_RAW_SEGMENT_CLASSES_9: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "flat"]
RID_RAW_SEGMENT_CLASSES_17: list[str] = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "flat",
]

_SEGMENT_SCHEMES: dict[int, list[str]] = {
    5: RID_RAW_SEGMENT_CLASSES_5,
    9: RID_RAW_SEGMENT_CLASSES_9,
    17: RID_RAW_SEGMENT_CLASSES_17,
}

# Compass bearing (degrees, 0=N clockwise) associated with each named direction. "flat" has no
# azimuth (a horizontal plane has no facing direction) and is handled specially by callers.
COMPASS_BEARINGS: dict[str, float] = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def get_segment_classes(scheme: int = 9) -> list[str]:
    if scheme not in _SEGMENT_SCHEMES:
        raise ValueError(f"Unsupported segment scheme '{scheme}'. Choose from {list(_SEGMENT_SCHEMES)}.")
    return _SEGMENT_SCHEMES[scheme]


def segment_classes_with_background(scheme: int = 9) -> list[str]:
    return ["background", *get_segment_classes(scheme)]


def azimuth_to_direction(azimuth_deg: float, scheme: int = 9) -> str:
    """Bin a continuous azimuth (degrees, 0=N clockwise) into the nearest named direction.

    Mirrors RID's own ``azimuth_to_label_class`` binning logic (equal-width sectors centered
    on each compass point), excluding the ``flat`` class which is not azimuth-derived.
    """
    directions = [d for d in get_segment_classes(scheme) if d != "flat"]
    n = len(directions)
    sector = 360.0 / n
    az = azimuth_deg % 360.0
    idx = int(((az + sector / 2) // sector) % n)
    return directions[idx]


@dataclass(frozen=True)
class ClassPalette:
    """Fixed RGB colors for consistent visualization across figures and the Streamlit app."""

    colors: dict[str, tuple[int, int, int]]

    def as_list(self, class_names: list[str]) -> list[tuple[int, int, int]]:
        return [self.colors.get(name, (255, 0, 255)) for name in class_names]


SUPERSTRUCTURE_PALETTE = ClassPalette({
    "background": (30, 30, 30),
    "pvmodule": (0, 128, 255),
    "dormer": (255, 165, 0),
    "window": (255, 255, 0),
    "ladder": (0, 255, 255),
    "chimney": (200, 0, 0),
    "shadow": (100, 100, 100),
    "tree": (0, 180, 0),
    "unknown": (255, 0, 255),
})

SEGMENT_PALETTE = ClassPalette({
    "background": (20, 20, 20),
    "N": (66, 133, 244), "NNE": (72, 150, 210), "NE": (78, 168, 178), "ENE": (84, 186, 146),
    "E": (90, 204, 114), "ESE": (130, 200, 90), "SE": (170, 196, 66), "SSE": (210, 192, 42),
    "S": (234, 67, 53), "SSW": (222, 90, 80), "SW": (210, 113, 107), "WSW": (198, 136, 134),
    "W": (186, 159, 161), "WNW": (160, 130, 190), "NW": (134, 101, 219), "NNW": (108, 72, 248),
    "flat": (150, 150, 150),
})
