from __future__ import annotations

import torch

from src.uncertainty.pipeline_uncertainty import build_uncertainty_report
from src.uncertainty.segmentation_uncertainty import compute_segmentation_uncertainty
from src.uncertainty.sensitivity import run_sensitivity_analysis


def test_sensitivity_analysis_constant_function_has_zero_spread() -> None:
    result = run_sensitivity_analysis(lambda m, c, g: 42.0, 0.3, 0.2, 0.1)
    assert result.relative_spread == 0.0
    assert result.uncertainty_level() == "Low"


def test_sensitivity_analysis_detects_variation() -> None:
    def fn(margin, clearance, gsd):
        return 100 - margin * 50 - clearance * 30

    result = run_sensitivity_analysis(fn, 0.3, 0.2, 0.1)
    assert result.high > result.low
    assert result.relative_spread > 0


def test_segmentation_uncertainty_confident_prediction_has_high_confidence() -> None:
    logits = torch.zeros(3, 4, 4)
    logits[0] = 10.0  # class 0 dominates everywhere -> high softmax margin
    result = compute_segmentation_uncertainty(logits)
    assert result.mean_confidence > 0.9
    assert result.low_confidence_fraction == 0.0


def test_segmentation_uncertainty_ambiguous_prediction_has_low_confidence() -> None:
    logits = torch.zeros(3, 4, 4)  # uniform logits -> uniform softmax -> near-zero margin
    result = compute_segmentation_uncertainty(logits)
    assert result.mean_confidence < 0.1


def test_uncertainty_report_widens_with_low_confidence() -> None:
    def fn(margin, clearance, gsd):
        return 20.0

    high_conf_report = build_uncertainty_report(fn, 0.3, 0.2, 0.1, mean_segmentation_confidence=0.95)
    low_conf_report = build_uncertainty_report(fn, 0.3, 0.2, 0.1, mean_segmentation_confidence=0.1)
    high_conf_range = high_conf_report.panel_count_high - high_conf_report.panel_count_low
    low_conf_range = low_conf_report.panel_count_high - low_conf_report.panel_count_low
    assert low_conf_range >= high_conf_range


def test_uncertainty_report_has_explanation_text() -> None:
    report = build_uncertainty_report(lambda m, c, g: 15.0, 0.3, 0.2, 0.1)
    assert "estimate" in report.explanation.lower()
    assert report.uncertainty_level in {"Low", "Medium", "High"}
