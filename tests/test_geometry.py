from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from src.geometry.orientation import estimate_orientation
from src.geometry.pitch import classify_roof_archetype, estimate_pitch
from src.geometry.polygons import mask_to_polygons, polygon_area_px, union_polygons
from src.geometry.roof_area import DEFAULT_GSD_M_PER_PX, estimate_area, pixel_area_from_mask
from src.geometry.usable_area import compute_usable_area, compute_usable_polygon


def test_mask_to_polygons_simple_square() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    polys = mask_to_polygons(mask, class_id=1)
    assert len(polys) == 1
    # cv2 contour tracing follows pixel boundaries (not pixel-area centers), so a filled NxN
    # block typically reports area close to (N-1)^2 rather than N^2 -- allow for that quirk.
    assert polygon_area_px(polys[0]) == pytest.approx(400, rel=0.15)


def test_mask_to_polygons_handles_hole() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    mask[25:35, 25:35] = 0  # hole
    polys = mask_to_polygons(mask, class_id=1)
    assert len(polys) == 1
    # area should be less than the full 40x40 square due to the hole
    assert polygon_area_px(polys[0]) < 40 * 40


def test_union_polygons_empty_list_returns_empty_polygon() -> None:
    result = union_polygons([])
    assert result.is_empty


def test_estimate_area_flags_assumed_gsd_when_none() -> None:
    area = estimate_area(pixel_area=1000, gsd_m_per_px=None)
    assert area.is_gsd_assumed is True
    assert area.gsd_m_per_px == DEFAULT_GSD_M_PER_PX
    assert area.area_m2 == pytest.approx(1000 * DEFAULT_GSD_M_PER_PX ** 2)


def test_estimate_area_with_explicit_gsd_not_flagged_assumed() -> None:
    area = estimate_area(pixel_area=1000, gsd_m_per_px=0.05)
    assert area.is_gsd_assumed is False
    assert area.area_m2 == pytest.approx(1000 * 0.05 ** 2)


def test_pixel_area_from_mask_counts_matching_classes() -> None:
    mask = np.array([[0, 1, 2], [1, 1, 0]])
    assert pixel_area_from_mask(mask, [1]) == 3
    assert pixel_area_from_mask(mask, [1, 2]) == 4


def test_usable_area_subtracts_obstacles_and_margin() -> None:
    roof = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    obstacle = Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
    result = compute_usable_area(
        roof, {"chimney": [obstacle]}, gsd_m_per_px=0.1, roof_edge_margin_m=0.5, obstacle_clearance_m=0.2,
    )
    assert result.usable_area.area_m2 < result.roof_area.area_m2
    assert result.obstacle_area.area_m2 > 0
    # usable polygon must not intersect the (buffered) obstacle
    assert result.usable_polygon.intersection(obstacle).area < obstacle.area


def test_usable_polygon_excludes_soft_classes_by_default() -> None:
    roof = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    shadow = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    usable, obstacle_union = compute_usable_polygon(roof, {"shadow": [shadow]}, roof_edge_margin_px=0, obstacle_clearance_px=0)
    assert obstacle_union.is_empty  # "shadow" is not in HARD_OBSTACLE_CLASSES
    assert usable.area == pytest.approx(roof.area)


def test_pitch_flat_roof_is_low_pitch() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 9  # "flat" class id in the 9-scheme (background + 8 dirs + flat)
    archetype = classify_roof_archetype(mask, segment_scheme=9)
    assert archetype == "flat"
    pitch = estimate_pitch(mask, segment_scheme=9)
    assert pitch.pitch_deg_typical < 10
    assert pitch.is_measurement is False


def test_pitch_gable_roof_has_higher_pitch_prior() -> None:
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[10:40, 10:50] = 1  # "N"
    mask[10:40, 50:90] = 5  # "S"
    archetype = classify_roof_archetype(mask, segment_scheme=9)
    assert archetype == "gable_or_hip"
    pitch = estimate_pitch(mask, segment_scheme=9)
    assert pitch.pitch_deg_typical > 20


def test_orientation_dominant_direction_matches_majority_class() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[:, :30] = 1  # "N" majority
    mask[:, 30:] = 5  # "S" minority
    result = estimate_orientation(mask, segment_scheme=9)
    assert result.dominant_direction == "N"
    assert not result.is_flat_dominant


def test_orientation_all_background_is_unknown() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    result = estimate_orientation(mask, segment_scheme=9)
    assert result.dominant_direction == "unknown"
