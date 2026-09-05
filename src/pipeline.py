"""End-to-end roof-analysis pipeline: segmentation -> geometry -> obstacles -> placement ->
capacity -> uncertainty. This is the single orchestration point used by ``scripts/predict.py``,
``scripts/run_experiment.py`` and ``app/streamlit_app.py`` so all three present identical logic.

Two entry modes are supported:

* **Model-driven** (``segment_model`` / ``superstructure_model`` given): runs real inference and
  derives a genuine segmentation-confidence signal for uncertainty.
* **Mask-driven** (``segment_mask`` / ``superstructure_mask`` given directly): used for the
  synthetic-ground-truth demo path and for testing, when no trained checkpoint is available yet.
  In this mode segmentation confidence is reported as unavailable rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from src.data.label_maps import (
    RID_RAW_SUPERSTRUCTURE_CLASSES,
    SUPERSTRUCTURE_CLASSES,
    segment_classes_with_background,
)
from src.data.transforms import build_eval_transforms
from src.geometry.orientation import (
    OrientationResult,
    direction_to_compass_label,
    estimate_orientation,
)
from src.geometry.pitch import PitchEstimate, estimate_pitch
from src.geometry.polygons import mask_to_polygons, union_polygons
from src.geometry.roof_area import DEFAULT_GSD_M_PER_PX
from src.geometry.usable_area import UsableAreaResult, compute_usable_area
from src.optimization.capacity import DEFAULT_PANEL_POWER_W, CapacityEstimate, estimate_capacity
from src.optimization.panel_placement import PlacementResult, optimize_placement_multi_orientation
from src.uncertainty.pipeline_uncertainty import UncertaintyReport, build_uncertainty_report
from src.uncertainty.segmentation_uncertainty import (
    SegmentationUncertaintyMap,
    compute_segmentation_uncertainty,
)


@dataclass
class SolarConfig:
    panel_width_m: float = 1.0
    panel_height_m: float = 1.7
    panel_power_w: float = DEFAULT_PANEL_POWER_W
    spacing_m: float = 0.02
    roof_edge_margin_m: float = 0.3
    obstacle_clearance_m: float = 0.2
    segment_scheme: int = 9


@dataclass
class PipelineResult:
    segment_mask: np.ndarray
    superstructure_mask: np.ndarray
    orientation: OrientationResult
    pitch: PitchEstimate
    usable: UsableAreaResult
    obstacle_polygons_by_class: dict[str, list]
    placement: PlacementResult
    capacity: CapacityEstimate
    uncertainty: UncertaintyReport
    seg_uncertainty: SegmentationUncertaintyMap | None
    gsd_m_per_px: float
    gsd_is_assumed: bool
    solar_config: SolarConfig = field(default_factory=SolarConfig)

    def report_lines(self) -> list[str]:
        gsd_flag = " (assumed)" if self.gsd_is_assumed else ""
        return [
            f"Roof area: {self.usable.roof_area.area_m2:.1f} m^2{gsd_flag} [{self.usable.roof_area.label()}]",
            f"Usable area: {self.usable.usable_area.area_m2:.1f} m^2{gsd_flag} [{self.usable.usable_area.label()}]",
            f"Obstacle area removed: {self.usable.obstacle_area.area_m2:.1f} m^2",
            f"Orientation: {direction_to_compass_label(self.orientation.dominant_direction)}",
            f"Estimated pitch: ~{self.pitch.pitch_deg_typical:.0f} deg "
            f"(archetype heuristic, range {self.pitch.pitch_deg_low:.0f}-{self.pitch.pitch_deg_high:.0f} deg -- NOT a measurement)",
            f"Panels placed: {len(self.placement.panels)} "
            f"(packing efficiency {self.placement.packing_efficiency * 100:.0f}%)",
            f"Panel rating: {self.solar_config.panel_power_w:.0f} W",
            f"Capacity: {self.capacity.capacity_kwp:.1f} kWp (theoretical installed capacity, not an energy-yield simulation)",
            f"Uncertainty range: {self.uncertainty.panel_count_low}-{self.uncertainty.panel_count_high} panels "
            f"({self.uncertainty.uncertainty_level})",
        ]


def _run_inference(model: torch.nn.Module, image: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    transform = build_eval_transforms(image_size)
    transformed = transform(image=image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
    tensor = transformed["image"].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    return logits.squeeze(0).cpu()


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def orientation_row_angle(orientation: OrientationResult) -> float:
    """Roof panel ROWS run parallel to the ridge, i.e. perpendicular to the slope-facing azimuth."""
    if orientation.is_flat_dominant or orientation.mean_azimuth_deg is None:
        return 0.0
    facing = orientation.mean_azimuth_deg
    # Image-space angle: compass bearing (0=N=up, clockwise) -> math angle used by panel grid
    # (0=+x axis). Rows run perpendicular to the facing direction.
    return (facing + 90.0) % 180.0


def run_pipeline(
    image: np.ndarray,
    solar_config: SolarConfig | None = None,
    gsd_m_per_px: float | None = None,
    segment_mask: np.ndarray | None = None,
    superstructure_mask: np.ndarray | None = None,
    segment_model: torch.nn.Module | None = None,
    superstructure_model: torch.nn.Module | None = None,
    model_image_size: int = 512,
    device: torch.device | None = None,
) -> PipelineResult:
    solar_config = solar_config or SolarConfig()
    gsd_is_assumed = gsd_m_per_px is None
    gsd = gsd_m_per_px if gsd_m_per_px is not None else DEFAULT_GSD_M_PER_PX

    h_orig, w_orig = image.shape[:2]
    seg_uncertainty: SegmentationUncertaintyMap | None = None

    if segment_mask is None:
        if segment_model is None:
            raise ValueError("Either segment_mask or segment_model must be provided.")
        device = device or next(segment_model.parameters()).device
        logits = _run_inference(segment_model, image, model_image_size, device)
        segment_mask = logits.argmax(dim=0).numpy().astype(np.uint8)
        segment_mask = _resize_mask(segment_mask, (h_orig, w_orig))
        roof_binary = segment_mask != 0
        seg_uncertainty = compute_segmentation_uncertainty(logits, roof_mask=_resize_mask(roof_binary.astype(np.uint8), (h_orig, w_orig)))

    if superstructure_mask is None:
        if superstructure_model is None:
            superstructure_mask = np.zeros_like(segment_mask)
        else:
            device = device or next(superstructure_model.parameters()).device
            logits = _run_inference(superstructure_model, image, model_image_size, device)
            superstructure_mask = logits.argmax(dim=0).numpy().astype(np.uint8)
            superstructure_mask = _resize_mask(superstructure_mask, (h_orig, w_orig))

    orientation = estimate_orientation(
        segment_mask, solar_config.segment_scheme, source="model_prediction" if segment_model else "ground_truth"
    )
    pitch = estimate_pitch(segment_mask, solar_config.segment_scheme)

    roof_polygons = mask_to_polygons(segment_mask, class_id=1)
    for cls_id in range(2, len(segment_classes_with_background(solar_config.segment_scheme))):
        roof_polygons.extend(mask_to_polygons(segment_mask, class_id=cls_id))
    roof_polygon = union_polygons(roof_polygons)

    obstacle_polygons_by_class: dict[str, list] = {}
    for cls_name in RID_RAW_SUPERSTRUCTURE_CLASSES:
        cls_id = SUPERSTRUCTURE_CLASSES.index(cls_name)
        obstacle_polygons_by_class[cls_name] = mask_to_polygons(superstructure_mask, cls_id)

    def _usable_for(margin_m: float, clearance_m: float, gsd_val: float) -> UsableAreaResult:
        return compute_usable_area(
            roof_polygon, obstacle_polygons_by_class, gsd_val, margin_m, clearance_m
        )

    usable = _usable_for(solar_config.roof_edge_margin_m, solar_config.obstacle_clearance_m, gsd)

    row_angle = orientation_row_angle(orientation)
    panel_w_px = solar_config.panel_width_m / gsd
    panel_h_px = solar_config.panel_height_m / gsd
    spacing_px = solar_config.spacing_m / gsd

    placement = optimize_placement_multi_orientation(
        usable.usable_polygon, panel_w_px, panel_h_px, spacing_px, base_orientation_deg=row_angle
    )

    capacity = estimate_capacity(len(placement.panels), solar_config.panel_power_w)

    def _panel_count_fn(margin_m: float, clearance_m: float, gsd_val: float) -> float:
        u = _usable_for(margin_m, clearance_m, gsd_val)
        pw = solar_config.panel_width_m / gsd_val
        ph = solar_config.panel_height_m / gsd_val
        sp = solar_config.spacing_m / gsd_val
        result = optimize_placement_multi_orientation(u.usable_polygon, pw, ph, sp, base_orientation_deg=row_angle)
        return float(len(result.panels))

    mean_conf = seg_uncertainty.mean_confidence if seg_uncertainty is not None else None
    uncertainty = build_uncertainty_report(
        _panel_count_fn,
        solar_config.roof_edge_margin_m,
        solar_config.obstacle_clearance_m,
        gsd,
        mean_segmentation_confidence=mean_conf,
    )

    return PipelineResult(
        segment_mask=segment_mask,
        superstructure_mask=superstructure_mask,
        orientation=orientation,
        pitch=pitch,
        usable=usable,
        obstacle_polygons_by_class=obstacle_polygons_by_class,
        placement=placement,
        capacity=capacity,
        uncertainty=uncertainty,
        seg_uncertainty=seg_uncertainty,
        gsd_m_per_px=gsd,
        gsd_is_assumed=gsd_is_assumed,
        solar_config=solar_config,
    )
