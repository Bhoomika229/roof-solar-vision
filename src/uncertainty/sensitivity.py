"""Sensitivity analysis: how much do downstream estimates move under plausible input perturbations?

This is the primary uncertainty signal for the *geometric/optimization* half of the pipeline
(panel count, usable area, capacity), where no model-probability signal exists. Rather than a
formal statistical confidence interval (which would require a calibrated error model we do not
have), we report an "uncertainty estimate": re-run the deterministic geometry/placement pipeline
under a small grid of realistic parameter perturbations (safety margin, obstacle clearance, GSD)
and report the resulting spread. This is honest about what it is — a sensitivity range, not a
confidence interval — per the project's uncertainty-quantification requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SensitivityResult:
    values: list[float]
    central_value: float
    low: float
    high: float
    relative_spread: float  # (high - low) / central_value

    def uncertainty_level(self) -> str:
        if not np.isfinite(self.relative_spread):
            return "Unknown"
        if self.relative_spread < 0.10:
            return "Low"
        if self.relative_spread < 0.25:
            return "Medium"
        return "High"


def run_sensitivity_analysis(
    evaluate_fn: Callable[[float, float, float], float],
    central_margin_m: float,
    central_clearance_m: float,
    central_gsd_m_per_px: float,
    margin_perturbation_frac: float = 0.3,
    clearance_perturbation_frac: float = 0.3,
    gsd_perturbation_frac: float = 0.05,
) -> SensitivityResult:
    """Evaluate ``evaluate_fn(margin_m, clearance_m, gsd_m_per_px) -> float`` at the central
    parameter setting plus every combination of +/- perturbation on each of the three inputs
    (a small full-factorial design: 3^3 = 27 evaluations), and summarize the output spread.
    """
    margins = [central_margin_m * f for f in (1 - margin_perturbation_frac, 1.0, 1 + margin_perturbation_frac)]
    clearances = [central_clearance_m * f for f in (1 - clearance_perturbation_frac, 1.0, 1 + clearance_perturbation_frac)]
    gsds = [central_gsd_m_per_px * f for f in (1 - gsd_perturbation_frac, 1.0, 1 + gsd_perturbation_frac)]

    values = [evaluate_fn(m, c, g) for m in margins for c in clearances for g in gsds]
    central_value = evaluate_fn(central_margin_m, central_clearance_m, central_gsd_m_per_px)

    low, high = float(np.min(values)), float(np.max(values))
    relative_spread = (high - low) / central_value if central_value > 0 else float("inf")

    return SensitivityResult(values=values, central_value=central_value, low=low, high=high, relative_spread=relative_spread)
