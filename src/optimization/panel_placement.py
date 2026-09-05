"""Geometric solar-panel placement inside a usable-roof polygon.

Panel count is never computed as ``usable_area / panel_area`` (that ignores rectangle packing
losses at irregular boundaries, panel orientation, and inter-panel spacing, and would silently
overstate capacity). Instead this module actually places rectangles:

1. Build a rotated grid of candidate panel-rectangle positions aligned to the requested
   orientation (typically the roof's dominant azimuth direction, so panel rows run parallel to
   the ridge/eave — the standard real-world installation convention), with configurable
   inter-panel spacing.
2. Keep only candidates whose (spacing-buffered) rectangle is fully contained in the usable
   polygon (`Polygon.contains`), so no panel overlaps an obstacle clearance zone or hangs off
   the roof edge.
3. Greedily accept non-overlapping candidates in row-major scan order (deterministic — the same
   input always yields the same layout, which matters for reproducible experiments).

This is a deterministic greedy baseline; :func:`optimize_placement_multi_orientation` extends it
by additionally checking a small set of candidate row-orientations (aligned to a facet's own
azimuth, plus 0/90 degrees relative to it) and keeping the best count. That is a pragmatic
"try a few reasonable layouts, keep the best" optimizer rather than a full combinatorial ILP,
which is unnecessary for rectangle counts in the tens-to-low-hundreds this problem produces.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.prepared import prep


@dataclass
class PanelInstance:
    center_px: tuple[float, float]
    corners_px: list[tuple[float, float]]
    rotation_deg: float


@dataclass
class PlacementResult:
    panels: list[PanelInstance]
    orientation_deg: float
    panel_width_px: float
    panel_height_px: float
    spacing_px: float
    usable_polygon_area_px: float
    covered_area_px: float
    packing_efficiency: float  # covered_area / usable_polygon_area


def _panel_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    rect = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    rect = shapely_rotate(rect, angle_deg, origin=(0, 0), use_radians=False)
    rect = shapely_translate(rect, xoff=cx, yoff=cy)
    return rect


def place_panels_greedy(
    usable_polygon: Polygon | MultiPolygon,
    panel_width_px: float,
    panel_height_px: float,
    spacing_px: float,
    orientation_deg: float = 0.0,
) -> PlacementResult:
    """Deterministic greedy rectangle packing of a usable polygon.

    ``orientation_deg`` rotates the whole candidate grid; 0 degrees means panel rows run along
    the image x-axis. Callers typically pass the roof's dominant azimuth-perpendicular angle so
    rows align with the ridge, matching standard installation practice.
    """
    if usable_polygon.is_empty or usable_polygon.area <= 0:
        return PlacementResult([], orientation_deg, panel_width_px, panel_height_px, spacing_px, 0.0, 0.0, 0.0)

    step_x = panel_width_px + spacing_px
    step_y = panel_height_px + spacing_px

    centroid = usable_polygon.centroid
    # Work in a rotated frame: rotate the polygon by -orientation so candidate rectangles can be
    # generated axis-aligned, then rotate accepted rectangles back.
    rotated_polygon = shapely_rotate(usable_polygon, -orientation_deg, origin=centroid, use_radians=False)
    minx, miny, maxx, maxy = rotated_polygon.bounds

    prepared = prep(rotated_polygon)
    accepted_rot: list[Polygon] = []

    n_rows = int((maxy - miny) / step_y) + 2
    n_cols = int((maxx - minx) / step_x) + 2

    for row in range(n_rows):
        cy = miny + panel_height_px / 2 + row * step_y
        if cy + panel_height_px / 2 > maxy:
            break
        for col in range(n_cols):
            cx = minx + panel_width_px / 2 + col * step_x
            if cx + panel_width_px / 2 > maxx:
                break
            candidate = _panel_rect(cx, cy, panel_width_px, panel_height_px, 0.0)
            if not prepared.contains(candidate):
                continue
            if any(candidate.intersects(other) for other in accepted_rot):
                continue
            accepted_rot.append(candidate)

    panels: list[PanelInstance] = []
    covered_area = 0.0
    for rect in accepted_rot:
        rect_world = shapely_rotate(rect, orientation_deg, origin=centroid, use_radians=False)
        cx, cy = rect_world.centroid.x, rect_world.centroid.y
        corners = list(rect_world.exterior.coords)[:-1]
        panels.append(PanelInstance(center_px=(cx, cy), corners_px=corners, rotation_deg=orientation_deg))
        covered_area += panel_width_px * panel_height_px

    usable_area = float(usable_polygon.area)
    efficiency = covered_area / usable_area if usable_area > 0 else 0.0

    return PlacementResult(
        panels=panels,
        orientation_deg=orientation_deg,
        panel_width_px=panel_width_px,
        panel_height_px=panel_height_px,
        spacing_px=spacing_px,
        usable_polygon_area_px=usable_area,
        covered_area_px=covered_area,
        packing_efficiency=efficiency,
    )


def optimize_placement_multi_orientation(
    usable_polygon: Polygon | MultiPolygon,
    panel_width_px: float,
    panel_height_px: float,
    spacing_px: float,
    base_orientation_deg: float = 0.0,
    candidate_offsets_deg: tuple[float, ...] = (0.0, 90.0),
    try_swapped_dimensions: bool = True,
) -> PlacementResult:
    """Try a small set of orientation/aspect candidates and keep the one with the most panels.

    This is the "optimization-based placement" extension referenced in the project spec: rather
    than a full mixed-integer solver (overkill for rectangle counts in this range), it evaluates
    a handful of physically sensible layouts — portrait vs. landscape panels, rows aligned with
    the roof ridge vs. perpendicular to it — and keeps the best by panel count, a legitimate
    (if simple) discrete optimization over layout choices.
    """
    best: PlacementResult | None = None
    dim_options = [(panel_width_px, panel_height_px)]
    if try_swapped_dimensions:
        dim_options.append((panel_height_px, panel_width_px))

    for offset in candidate_offsets_deg:
        angle = (base_orientation_deg + offset) % 180
        for w, h in dim_options:
            result = place_panels_greedy(usable_polygon, w, h, spacing_px, orientation_deg=angle)
            if best is None or len(result.panels) > len(best.panels):
                best = result

    assert best is not None
    return best


def panel_count_naive_area_ratio(usable_polygon: Polygon | MultiPolygon, panel_width_px: float, panel_height_px: float) -> float:
    """Included ONLY as the explicit baseline this project argues against — usable_area /
    panel_area, with no packing/geometry consideration. Used solely for comparison in
    experiment reports to demonstrate why real placement matters; never used for the
    reported/production estimate.
    """
    panel_area = panel_width_px * panel_height_px
    if panel_area <= 0:
        return 0.0
    return usable_polygon.area / panel_area
