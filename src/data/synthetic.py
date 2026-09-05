"""Synthetic aerial-roof-image generator.

RID cannot be redistributed or auto-downloaded here (see ``data/README.md``), so this module
provides a procedural stand-in that lets the *entire* non-training pipeline (geometry,
obstacle handling, panel placement, capacity, uncertainty, visualization, the Streamlit demo,
and the unit tests) run without any external dataset.

Unlike real monocular aerial imagery, the synthetic generator *knows* the true 3D roof pitch
and azimuth it used to render each sample (they are generation parameters, not measurements).
That makes synthetic data useful for validating the geometry/optimization code against known
ground truth — it must never be confused with evidence that pitch is recoverable from a real
single nadir photograph (see ``CLAUDE.md`` and ``docs/research_report.md`` § Limitations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.data.label_maps import (
    RID_RAW_SUPERSTRUCTURE_CLASSES,
    azimuth_to_direction,
    segment_classes_with_background,
)

BACKGROUND_ID = 0


@dataclass
class SyntheticSample:
    sample_id: str
    image: np.ndarray  # (H, W, 3) uint8
    segment_mask: np.ndarray  # (H, W) uint8, class ids into segment_classes_with_background()
    superstructure_mask: np.ndarray  # (H, W) uint8, class ids into SUPERSTRUCTURE_CLASSES
    metadata: dict[str, Any] = field(default_factory=dict)


def _rotate_point(pt: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return (pt - center) @ r.T + center


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


class SyntheticRoofGenerator:
    """Procedurally generates (image, segment_mask, superstructure_mask, metadata) triples.

    Archetypes: ``flat`` (single horizontal plane), ``gable`` (two opposing pitched planes),
    ``hip`` (four pitched planes around a central ridge point). Obstacles are drawn from RID's
    real superstructure taxonomy (see ``label_maps.py``).
    """

    def __init__(
        self,
        image_size: int = 512,
        segment_scheme: int = 9,
        gsd_m_per_px: float = 0.10,
        seed: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.segment_scheme = segment_scheme
        self.gsd_m_per_px = gsd_m_per_px
        self.rng = np.random.default_rng(seed)
        self.segment_classes = segment_classes_with_background(segment_scheme)

    def _seg_id(self, name: str) -> int:
        return self.segment_classes.index(name)

    def _obstacle_id(self, name: str) -> int:
        return ["background", *RID_RAW_SUPERSTRUCTURE_CLASSES].index(name)

    # ------------------------------------------------------------------
    def generate(self, sample_id: str) -> SyntheticSample:
        rng = self.rng
        size = self.image_size
        image = self._make_background(size)
        segment_mask = np.zeros((size, size), dtype=np.uint8)
        superstructure_mask = np.zeros((size, size), dtype=np.uint8)

        archetype = rng.choice(["flat", "gable", "hip"], p=[0.25, 0.5, 0.25])
        center = np.array([size / 2, size / 2]) + rng.uniform(-size * 0.05, size * 0.05, size=2)
        half_w = rng.uniform(size * 0.22, size * 0.36)
        half_h = rng.uniform(size * 0.22, size * 0.36)
        rotation = float(rng.uniform(0, 180))

        footprint = np.array([
            center + [-half_w, -half_h],
            center + [half_w, -half_h],
            center + [half_w, half_h],
            center + [-half_w, half_h],
        ])
        footprint = np.array([_rotate_point(p, center, rotation) for p in footprint])
        footprint = np.clip(footprint, 4, size - 4)

        segments_meta: list[dict[str, Any]] = []
        base_color = tuple(int(c) for c in rng.integers(90, 150, size=3))

        if archetype == "flat":
            self._fill_poly(segment_mask, footprint, self._seg_id("flat"))
            self._fill_poly(image, footprint, base_color)
            segments_meta.append({
                "direction": "flat", "azimuth_deg": None, "slope_deg": float(rng.uniform(0, 3)),
                "area_px": float(cv2.contourArea(footprint.astype(np.float32))),
            })
        elif archetype == "gable":
            mid_a = (footprint[0] + footprint[1]) / 2
            mid_b = (footprint[2] + footprint[3]) / 2
            poly_a = np.array([footprint[0], footprint[1], mid_b, mid_a])
            poly_b = np.array([mid_a, mid_b, footprint[2], footprint[3]])
            slope = float(rng.uniform(20, 45))
            az_a = rotation % 360
            az_b = (rotation + 180) % 360
            for poly, az, shade in ((poly_a, az_a, 1.08), (poly_b, az_b, 0.85)):
                direction = azimuth_to_direction(az, self.segment_scheme)
                self._fill_poly(segment_mask, poly, self._seg_id(direction))
                self._fill_poly(image, poly, _shade(base_color, shade))
                segments_meta.append({
                    "direction": direction, "azimuth_deg": float(az), "slope_deg": slope,
                    "area_px": float(cv2.contourArea(poly.astype(np.float32))),
                })
            self._draw_line(image, mid_a, mid_b, (40, 40, 40), 2)
        else:  # hip
            apex = center + rng.uniform(-size * 0.05, size * 0.05, size=2)
            slope = float(rng.uniform(20, 40))
            shades = [1.1, 0.95, 0.8, 1.0]
            for i in range(4):
                p1, p2 = footprint[i], footprint[(i + 1) % 4]
                poly = np.array([p1, p2, apex])
                edge_mid = (p1 + p2) / 2
                az = (math.degrees(math.atan2(edge_mid[0] - center[0], -(edge_mid[1] - center[1]))) + rotation) % 360
                direction = azimuth_to_direction(az, self.segment_scheme)
                self._fill_poly(segment_mask, poly, self._seg_id(direction))
                self._fill_poly(image, poly, _shade(base_color, shades[i]))
                segments_meta.append({
                    "direction": direction, "azimuth_deg": float(az), "slope_deg": slope,
                    "area_px": float(cv2.contourArea(poly.astype(np.float32))),
                })
            for i in range(4):
                self._draw_line(image, footprint[i], apex, (40, 40, 40), 1)

        image = self._add_noise(image)

        obstacles_meta = self._add_obstacles(image, superstructure_mask, footprint, center, rng)

        metadata = {
            "sample_id": sample_id,
            "image_size": size,
            "gsd_m_per_px": self.gsd_m_per_px,
            "archetype": archetype,
            "roof_footprint_px": footprint.tolist(),
            "rotation_deg": rotation,
            "segments": segments_meta,
            "obstacles": obstacles_meta,
            "segment_scheme": self.segment_scheme,
            "note": (
                "slope_deg / azimuth_deg are GENERATION ground truth for this synthetic sample "
                "(known because we procedurally created the roof); they are not derived from "
                "the rendered pixels and must not be treated as evidence that pitch is directly "
                "measurable from a single real aerial photo."
            ),
        }
        return SyntheticSample(sample_id, image, segment_mask, superstructure_mask, metadata)

    # ------------------------------------------------------------------
    def _make_background(self, size: int) -> np.ndarray:
        rng = self.rng
        base = rng.integers(60, 100)
        image = np.full((size, size, 3), base, dtype=np.int16)
        noise = rng.normal(0, 6, size=(size, size, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        # a few procedural "vegetation" patches so the scene doesn't look like flat noise
        for _ in range(rng.integers(2, 6)):
            c = rng.uniform(0, size, size=2)
            r = rng.uniform(size * 0.03, size * 0.09)
            color = (int(rng.integers(20, 60)), int(rng.integers(70, 120)), int(rng.integers(20, 60)))
            cv2.circle(image, tuple(c.astype(int)), int(r), color, -1)
        return image

    @staticmethod
    def _fill_poly(canvas: np.ndarray, poly: np.ndarray, value: Any) -> None:
        pts = poly.astype(np.int32).reshape(1, -1, 2)
        if canvas.ndim == 2:
            cv2.fillPoly(canvas, pts, int(value))
        else:
            cv2.fillPoly(canvas, pts, tuple(int(v) for v in value))

    @staticmethod
    def _draw_line(canvas: np.ndarray, p1: np.ndarray, p2: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, 4, size=image.shape)
        out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (3, 3), 0)

    def _add_obstacles(
        self,
        image: np.ndarray,
        superstructure_mask: np.ndarray,
        footprint: np.ndarray,
        center: np.ndarray,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        from src.data.label_maps import SUPERSTRUCTURE_PALETTE

        size = self.image_size
        min_xy = footprint.min(axis=0)
        max_xy = footprint.max(axis=0)
        obstacles: list[dict[str, Any]] = []

        def random_point_in_footprint(margin: float) -> np.ndarray:
            for _ in range(20):
                p = rng.uniform(min_xy + margin, max_xy - margin)
                if cv2.pointPolygonTest(footprint.astype(np.float32), tuple(p), False) >= 0:
                    return p
            return center

        n_chimneys = rng.integers(0, 3)
        for _ in range(n_chimneys):
            p = random_point_in_footprint(size * 0.08)
            w, h = rng.uniform(6, 14), rng.uniform(6, 14)
            poly = np.array([p + [-w, -h], p + [w, -h], p + [w, h], p + [-w, h]])
            self._paint_obstacle(image, superstructure_mask, poly, "chimney", SUPERSTRUCTURE_PALETTE)
            obstacles.append({"class": "chimney", "polygon_px": poly.tolist()})

        if rng.random() < 0.4:
            p = random_point_in_footprint(size * 0.1)
            w, h = rng.uniform(20, 35), rng.uniform(15, 25)
            poly = np.array([p + [-w, -h], p + [w, -h], p + [w, h], p + [-w, h]])
            self._paint_obstacle(image, superstructure_mask, poly, "dormer", SUPERSTRUCTURE_PALETTE)
            obstacles.append({"class": "dormer", "polygon_px": poly.tolist()})

        n_windows = rng.integers(0, 3)
        for _ in range(n_windows):
            p = random_point_in_footprint(size * 0.06)
            w, h = rng.uniform(8, 14), rng.uniform(8, 14)
            poly = np.array([p + [-w, -h], p + [w, -h], p + [w, h], p + [-w, h]])
            self._paint_obstacle(image, superstructure_mask, poly, "window", SUPERSTRUCTURE_PALETTE)
            obstacles.append({"class": "window", "polygon_px": poly.tolist()})

        if rng.random() < 0.3:
            p1 = random_point_in_footprint(size * 0.05)
            angle = rng.uniform(0, 360)
            length = rng.uniform(30, 60)
            p2 = p1 + length * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            thickness = 3
            color = SUPERSTRUCTURE_PALETTE.colors["ladder"]
            cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness)
            cv2.line(superstructure_mask, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     self._obstacle_id("ladder"), thickness)
            obstacles.append({"class": "ladder", "polygon_px": [p1.tolist(), p2.tolist()]})

        if rng.random() < 0.35:
            n_rows, n_cols = int(rng.integers(1, 3)), int(rng.integers(2, 5))
            p = random_point_in_footprint(size * 0.15)
            pw, ph, gap = 14, 22, 2
            for r in range(n_rows):
                for c in range(n_cols):
                    corner = p + np.array([c * (pw + gap), r * (ph + gap)])
                    poly = np.array([corner, corner + [pw, 0], corner + [pw, ph], corner + [0, ph]])
                    if cv2.pointPolygonTest(footprint.astype(np.float32), tuple(poly.mean(axis=0)), False) >= 0:
                        self._paint_obstacle(image, superstructure_mask, poly, "pvmodule", SUPERSTRUCTURE_PALETTE)
                        obstacles.append({"class": "pvmodule", "polygon_px": poly.tolist()})

        if rng.random() < 0.3:
            tree_center = center + rng.uniform(-size * 0.4, size * 0.4, size=2)
            tree_r = rng.uniform(size * 0.04, size * 0.09)
            cv2.circle(image, tuple(tree_center.astype(int)), int(tree_r), (20, 90, 20), -1)
            shadow_dir = rng.uniform(0, 2 * math.pi)
            shadow_end = tree_center + tree_r * 2.2 * np.array([math.cos(shadow_dir), math.sin(shadow_dir)])
            shadow_poly = np.array([
                tree_center + [tree_r * 0.6, 0], tree_center + [-tree_r * 0.6, 0],
                shadow_end + [-tree_r * 0.3, 0], shadow_end + [tree_r * 0.3, 0],
            ])
            clipped = self._clip_to_footprint(shadow_poly, footprint)
            if clipped is not None and len(clipped) >= 3:
                self._paint_obstacle(image, superstructure_mask, clipped, "shadow", SUPERSTRUCTURE_PALETTE, alpha=0.35)
                obstacles.append({"class": "shadow", "polygon_px": clipped.tolist()})
            if cv2.pointPolygonTest(footprint.astype(np.float32), tuple(tree_center), False) >= 0:
                r = int(tree_r)
                cv2.circle(superstructure_mask, tuple(tree_center.astype(int)), r, self._obstacle_id("tree"), -1)
                obstacles.append({"class": "tree", "polygon_px": None})

        return obstacles

    @staticmethod
    def _clip_to_footprint(poly: np.ndarray, footprint: np.ndarray) -> np.ndarray | None:
        try:
            from shapely.geometry import Polygon

            a = Polygon(poly).buffer(0)
            b = Polygon(footprint).buffer(0)
            inter = a.intersection(b)
            if inter.is_empty or inter.area < 4:
                return None
            if inter.geom_type == "Polygon":
                return np.array(inter.exterior.coords)
            largest = max(inter.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)
        except Exception:
            return None

    @staticmethod
    def _paint_obstacle(
        image: np.ndarray,
        mask: np.ndarray,
        poly: np.ndarray,
        class_name: str,
        palette,
        alpha: float = 0.75,
    ) -> None:
        color = palette.colors[class_name]
        overlay = image.copy()
        SyntheticRoofGenerator._fill_poly(overlay, poly, color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
        class_names = ["background", *RID_RAW_SUPERSTRUCTURE_CLASSES]
        SyntheticRoofGenerator._fill_poly(mask, poly, class_names.index(class_name))
