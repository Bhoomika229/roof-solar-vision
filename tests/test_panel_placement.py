from __future__ import annotations

from shapely.geometry import Polygon

from src.optimization.panel_placement import (
    optimize_placement_multi_orientation,
    panel_count_naive_area_ratio,
    place_panels_greedy,
)


def test_place_panels_in_large_square() -> None:
    roof = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    result = place_panels_greedy(roof, panel_width_px=100, panel_height_px=170, spacing_px=5)
    assert len(result.panels) > 0
    assert result.packing_efficiency > 0
    assert result.packing_efficiency <= 1.0001


def test_placed_panels_do_not_overlap() -> None:
    roof = Polygon([(0, 0), (500, 0), (500, 500), (0, 500)])
    result = place_panels_greedy(roof, panel_width_px=80, panel_height_px=140, spacing_px=4)
    corners = [p.corners_px for p in result.panels]
    polys = [Polygon(c) for c in corners]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            assert polys[i].intersection(polys[j]).area < 1e-6


def test_placed_panels_are_contained_in_roof_polygon() -> None:
    roof = Polygon([(0, 0), (400, 0), (400, 400), (0, 400)])
    result = place_panels_greedy(roof, panel_width_px=90, panel_height_px=150, spacing_px=5)
    for panel in result.panels:
        panel_poly = Polygon(panel.corners_px)
        # allow tiny floating point slack
        assert roof.buffer(1e-6).contains(panel_poly)


def test_empty_polygon_yields_no_panels() -> None:
    result = place_panels_greedy(Polygon(), panel_width_px=100, panel_height_px=170, spacing_px=5)
    assert result.panels == []
    assert result.packing_efficiency == 0.0


def test_too_small_polygon_yields_no_panels() -> None:
    tiny = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    result = place_panels_greedy(tiny, panel_width_px=100, panel_height_px=170, spacing_px=5)
    assert result.panels == []


def test_optimize_multi_orientation_is_at_least_as_good_as_single() -> None:
    # A long thin roof strongly favors one orientation over its perpendicular -- multi-orientation
    # search must find a layout at least as good as any single fixed orientation.
    roof = Polygon([(0, 0), (1000, 0), (1000, 150), (0, 150)])
    single = place_panels_greedy(roof, panel_width_px=100, panel_height_px=170, spacing_px=5, orientation_deg=0)
    best = optimize_placement_multi_orientation(roof, panel_width_px=100, panel_height_px=170, spacing_px=5)
    assert len(best.panels) >= len(single.panels)


def test_naive_ratio_overstates_relative_to_real_placement() -> None:
    """Demonstrates why panel count must come from real placement, not area / panel_area:
    the naive ratio ignores packing losses at irregular/non-multiple-of-panel boundaries and
    so is >= the real placement count for a non-trivial polygon."""
    roof = Polygon([(0, 0), (537, 0), (537, 481), (0, 481)])  # deliberately awkward dimensions
    real = place_panels_greedy(roof, panel_width_px=100, panel_height_px=170, spacing_px=10)
    naive = panel_count_naive_area_ratio(roof, panel_width_px=100, panel_height_px=170)
    assert naive >= len(real.panels)
