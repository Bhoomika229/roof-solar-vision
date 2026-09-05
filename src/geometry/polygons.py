"""Raster mask <-> vector polygon conversion utilities, built on OpenCV contours + Shapely.

All downstream geometric reasoning (area, usable-area masking, panel placement) operates on
Shapely polygons rather than raw pixel masks, since boolean set operations (difference, buffer,
intersection) are what "usable area = roof - obstacles - margins" actually requires.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


def mask_to_polygons(mask: np.ndarray, class_id: int, min_area_px: float = 4.0) -> list[Polygon]:
    """Extract one Shapely polygon per connected component of ``mask == class_id``.

    Uses OpenCV's ``RETR_CCOMP`` contour hierarchy so holes (e.g. a chimney fully inside a roof
    polygon) are represented as interior rings rather than being ignored or merged away.
    """
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    polygons: list[Polygon] = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] != -1:
            continue  # this contour is a hole, handled via its parent below
        if len(contour) < 3:
            continue
        exterior = contour.squeeze(1)
        holes = []
        child_idx = hierarchy[idx][2]
        while child_idx != -1:
            child_contour = contours[child_idx].squeeze(1)
            if len(child_contour) >= 3:
                holes.append(child_contour)
            child_idx = hierarchy[child_idx][0]
        try:
            poly = Polygon(exterior, holes)
            poly = make_valid(poly) if not poly.is_valid else poly
        except Exception:
            continue
        if poly.area >= min_area_px:
            polygons.append(poly)

    return polygons


def polygons_from_multi_classes(mask: np.ndarray, class_ids: list[int], min_area_px: float = 4.0) -> list[Polygon]:
    polys: list[Polygon] = []
    for cid in class_ids:
        polys.extend(mask_to_polygons(mask, cid, min_area_px))
    return polys


def union_polygons(polygons: list[Polygon]) -> Polygon | MultiPolygon:
    if not polygons:
        return Polygon()
    return unary_union(polygons)


def polygon_area_px(polygon: Polygon | MultiPolygon) -> float:
    return float(polygon.area)


def simplify_polygon(polygon: Polygon, tolerance_px: float = 1.5) -> Polygon:
    return polygon.simplify(tolerance_px, preserve_topology=True)


def erode_polygon(polygon: Polygon | MultiPolygon, distance_px: float) -> Polygon | MultiPolygon:
    """Shrink a polygon inward by ``distance_px`` (safety-margin buffer)."""
    if polygon.is_empty or distance_px <= 0:
        return polygon
    return polygon.buffer(-distance_px)


def polygon_to_pixel_coords(polygon: Polygon) -> list[list[float]]:
    if polygon.is_empty:
        return []
    return [list(coord) for coord in polygon.exterior.coords]
