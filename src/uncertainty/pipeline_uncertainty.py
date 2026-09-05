"""Combine segmentation confidence + geometric sensitivity into one uncertainty report.

Final panel-count range = sensitivity-analysis range over safety-margin/clearance/GSD,
additionally widened when segmentation confidence is low (the geometry can only be as reliable
as the mask it was computed from). The widening factor is a documented, simple heuristic:
    widened_half_range = half_range * (1 + (1 - mean_confidence))
i.e. at perfect confidence (mean_confidence=1) the range is unchanged; at very low confidence
(mean_confidence -> 0) the range roughly doubles. This is a transparent, inspectable formula,
not a black-box or random number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.uncertainty.sensitivity import SensitivityResult, run_sensitivity_analysis


@dataclass
class UncertaintyReport:
    panel_count_central: int
    panel_count_low: int
    panel_count_high: int
    uncertainty_level: str
    mean_segmentation_confidence: float | None
    explanation: str


def build_uncertainty_report(
    panel_count_fn: Callable[[float, float, float], float],
    central_margin_m: float,
    central_clearance_m: float,
    central_gsd_m_per_px: float,
    mean_segmentation_confidence: float | None = None,
) -> UncertaintyReport:
    sensitivity: SensitivityResult = run_sensitivity_analysis(
        panel_count_fn, central_margin_m, central_clearance_m, central_gsd_m_per_px
    )

    half_range = (sensitivity.high - sensitivity.low) / 2
    center = sensitivity.central_value

    if mean_segmentation_confidence is not None and mean_segmentation_confidence == mean_segmentation_confidence:
        widen_factor = 1 + (1 - mean_segmentation_confidence)
        half_range *= widen_factor

    low = max(0, round(center - half_range))
    high = round(center + half_range)
    central = round(center)

    relative_spread = (high - low) / central if central > 0 else float("inf")
    if relative_spread < 0.10:
        level = "Low"
    elif relative_spread < 0.25:
        level = "Medium"
    else:
        level = "High"

    explanation = (
        f"Range computed by re-running panel placement under +/-30% perturbations of the "
        f"safety margin and obstacle clearance and +/-5% of GSD (a {'3^3=27' } full-factorial "
        f"sensitivity sweep), "
        + (
            f"then widened by a factor of {1 + (1 - mean_segmentation_confidence):.2f}x because the "
            f"mean segmentation confidence for this prediction was {mean_segmentation_confidence:.2f}. "
            if mean_segmentation_confidence is not None and mean_segmentation_confidence == mean_segmentation_confidence
            else ""
        )
        + "This is an uncertainty ESTIMATE derived from the pipeline's own sensitivity, not a "
        "formally calibrated statistical confidence interval."
    )

    return UncertaintyReport(
        panel_count_central=central,
        panel_count_low=low,
        panel_count_high=high,
        uncertainty_level=level,
        mean_segmentation_confidence=mean_segmentation_confidence,
        explanation=explanation,
    )
