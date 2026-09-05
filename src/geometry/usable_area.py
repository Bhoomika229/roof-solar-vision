"""Usable-roof-area computation via real geometric masking (not a flat percentage subtraction).

    usable_polygon = (roof_polygon  -buffer->  eroded by safety margin)
                      MINUS (each obstacle polygon, itself buffered outward by its own margin)

Both erosion (roof edges are unsafe for panel mounting near the eave/ridge/hip) and obstacle
dilation (a chimney needs clearance around it, not just its own footprint) are applied because
that is how real PV-installation planning actually works, and because it is straightforward to
justify geometrically — unlike an arbitrary "subtract 20% of area" rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from src.data.label_maps import HARD_OBSTACLE_CLASSES
from src.geometry.roof_area import AreaEstimate, estimate_area


@dataclass
class UsableAreaResult:
    roof_area: AreaEstimate
    obstacle_area: AreaEstimate
    usable_area: AreaEstimate
    usable_polygon: Polygon | MultiPolygon
    roof_edge_margin_px: float
    obstacle_clearance_px: float


def compute_usable_polygon(
    roof_polygon: Polygon | MultiPolygon,
    obstacle_polygons_by_class: dict[str, list[Polygon]],
    roof_edge_margin_px: float,
    obstacle_clearance_px: float,
    obstacle_classes_to_exclude: set[str] | None = None,
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Return ``(usable_polygon, obstacle_union_polygon)`` in pixel coordinates.

    ``obstacle_classes_to_exclude`` defaults to :data:`HARD_OBSTACLE_CLASSES` (chimneys,
    dormers, windows, ladders, trees, unknown) — soft classes like "shadow" and the
    "pvmodule" class (already-installed panels, not something to avoid but to report
    separately) are not treated as hard obstacles unless explicitly requested.
    """
    exclude = obstacle_classes_to_exclude or HARD_OBSTACLE_CLASSES

    eroded_roof = roof_polygon.buffer(-roof_edge_margin_px) if roof_edge_margin_px > 0 else roof_polygon
    if eroded_roof.is_empty:
        return eroded_roof, Polygon()

    obstacle_polys: list[Polygon] = []
    for class_name, polys in obstacle_polygons_by_class.items():
        if class_name not in exclude:
            continue
        for poly in polys:
            buffered = poly.buffer(obstacle_clearance_px) if obstacle_clearance_px > 0 else poly
            obstacle_polys.append(buffered)

    obstacle_union = unary_union(obstacle_polys) if obstacle_polys else Polygon()
    usable = eroded_roof.difference(obstacle_union) if not obstacle_union.is_empty else eroded_roof
    return usable, obstacle_union


def compute_usable_area(
    roof_polygon: Polygon | MultiPolygon,
    obstacle_polygons_by_class: dict[str, list[Polygon]],
    gsd_m_per_px: float | None,
    roof_edge_margin_m: float = 0.3,
    obstacle_clearance_m: float = 0.2,
) -> UsableAreaResult:
    from src.geometry.roof_area import DEFAULT_GSD_M_PER_PX

    gsd = gsd_m_per_px if gsd_m_per_px is not None else DEFAULT_GSD_M_PER_PX
    margin_px = roof_edge_margin_m / gsd
    clearance_px = obstacle_clearance_m / gsd

    usable_polygon, obstacle_union = compute_usable_polygon(
        roof_polygon, obstacle_polygons_by_class, margin_px, clearance_px
    )

    roof_area = estimate_area(float(roof_polygon.area), gsd_m_per_px)
    obstacle_area = estimate_area(float(obstacle_union.area), gsd_m_per_px)
    usable_area = estimate_area(float(usable_polygon.area), gsd_m_per_px)

    return UsableAreaResult(
        roof_area=roof_area,
        obstacle_area=obstacle_area,
        usable_area=usable_area,
        usable_polygon=usable_polygon,
        roof_edge_margin_px=margin_px,
        obstacle_clearance_px=clearance_px,
    )
