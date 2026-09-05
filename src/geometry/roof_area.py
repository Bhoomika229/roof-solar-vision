"""Physical roof-area estimation from a pixel mask + Ground Sampling Distance (GSD).

``physical_area_m2 = pixel_area * gsd_m_per_px ** 2``

GSD (metres of ground distance per image pixel) is a property of the *acquisition*, not
something derivable from the pixels themselves. RID's own imagery is roof-centered aerial
photography without a documented per-image GSD in the public repository metadata, so this
project treats GSD as a required, user-supplied (or config-default) parameter and always labels
any area computed with an assumed/default GSD as an ESTIMATE, never as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

DEFAULT_GSD_M_PER_PX = 0.10  # documented assumption: typical high-res aerial/orthophoto GSD


@dataclass
class AreaEstimate:
    pixel_area: float
    gsd_m_per_px: float
    area_m2: float
    is_gsd_assumed: bool

    def label(self) -> str:
        return "estimated (assumed GSD)" if self.is_gsd_assumed else "estimated (from provided GSD)"


def pixel_area_from_mask(mask: np.ndarray, class_ids: list[int]) -> float:
    return float(np.isin(mask, class_ids).sum())


def estimate_area(
    pixel_area: float,
    gsd_m_per_px: float | None,
    is_gsd_assumed: bool | None = None,
) -> AreaEstimate:
    """Convert a pixel-space area to physical m^2.

    If ``gsd_m_per_px`` is None, :data:`DEFAULT_GSD_M_PER_PX` is used and the result is always
    flagged ``is_gsd_assumed=True`` so callers cannot silently present it as measured.
    """
    assumed = is_gsd_assumed if is_gsd_assumed is not None else gsd_m_per_px is None
    gsd = gsd_m_per_px if gsd_m_per_px is not None else DEFAULT_GSD_M_PER_PX
    area_m2 = pixel_area * (gsd ** 2)
    return AreaEstimate(pixel_area=pixel_area, gsd_m_per_px=gsd, area_m2=area_m2, is_gsd_assumed=assumed)


def polygon_area_estimate(polygon: Polygon | MultiPolygon, gsd_m_per_px: float | None) -> AreaEstimate:
    return estimate_area(float(polygon.area), gsd_m_per_px)
