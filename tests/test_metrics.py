from __future__ import annotations

import pytest
import torch

from src.evaluation.metrics import SegmentationMetrics


def test_perfect_prediction_gives_iou_one() -> None:
    metrics = SegmentationMetrics(num_classes=3)
    target = torch.tensor([[0, 1, 2], [1, 1, 0]])
    metrics.update(target.clone(), target.clone())
    summary = metrics.compute()
    assert summary["mean_iou"] == pytest.approx(1.0)
    assert summary["pixel_accuracy"] == pytest.approx(1.0)


def test_completely_wrong_prediction_gives_zero_iou_for_present_classes() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    target = torch.zeros((4, 4), dtype=torch.long)
    pred = torch.ones((4, 4), dtype=torch.long)
    metrics.update(pred, target)
    summary = metrics.compute()
    assert summary["per_class_iou"][0] == 0.0
    assert summary["pixel_accuracy"] == 0.0


def test_metrics_accumulate_across_batches() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([[0, 0], [1, 1]]), torch.tensor([[0, 0], [1, 1]]))
    metrics.update(torch.tensor([[1, 1], [0, 0]]), torch.tensor([[0, 0], [1, 1]]))
    summary = metrics.compute()
    assert summary["pixel_accuracy"] == pytest.approx(0.5)


def test_compute_named_matches_class_list() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    named = metrics.compute_named(["background", "roof"])
    assert set(named["per_class"].keys()) == {"background", "roof"}
    assert named["overall"]["mean_iou"] == pytest.approx(1.0)


def test_reset_clears_confusion_matrix() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    metrics.reset()
    assert metrics.confusion.sum() == 0
