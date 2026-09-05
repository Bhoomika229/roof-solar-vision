from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset import (
    DatasetDiscoveryError,
    RoofDataset,
    discover_dataset_root,
    list_sample_ids,
)
from src.data.splits import load_splits, make_splits, save_splits
from src.data.transforms import build_eval_transforms


def test_discover_dataset_root(synthetic_dataset_dir: Path) -> None:
    layout = discover_dataset_root(synthetic_dataset_dir)
    assert layout["images"].name == "images"
    assert layout["segments"].name == "masks_segments"
    assert layout["superstructures"].name == "masks_superstructures"


def test_discover_missing_root_raises() -> None:
    with pytest.raises(DatasetDiscoveryError):
        discover_dataset_root("this/path/does/not/exist")


def test_list_sample_ids(synthetic_dataset_dir: Path) -> None:
    ids = list_sample_ids(synthetic_dataset_dir)
    assert len(ids) == 10
    assert ids == sorted(ids)


def test_roof_dataset_getitem_no_transform(synthetic_dataset_dir: Path) -> None:
    ds = RoofDataset(synthetic_dataset_dir, transform=None)
    item = ds[0]
    assert item["image"].shape[0] == 3
    assert item["segment_mask"].ndim == 2
    assert item["superstructure_mask"].shape == item["segment_mask"].shape
    assert "gsd_m_per_px" in item["metadata"]


def test_roof_dataset_with_transform_resizes(synthetic_dataset_dir: Path) -> None:
    transform = build_eval_transforms(image_size=64)
    ds = RoofDataset(synthetic_dataset_dir, transform=transform)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["segment_mask"].shape == (64, 64)


def test_make_and_load_splits(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    splits = make_splits(synthetic_dataset_dir, train_frac=0.6, val_frac=0.2, seed=1)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 10
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_ids) == 10  # no duplicates / omissions

    splits_dir = tmp_path / "splits"
    save_splits(splits, splits_dir)
    loaded = load_splits(splits_dir)
    assert loaded == splits


def test_splits_are_reproducible(synthetic_dataset_dir: Path) -> None:
    s1 = make_splits(synthetic_dataset_dir, seed=7)
    s2 = make_splits(synthetic_dataset_dir, seed=7)
    assert s1 == s2

    s3 = make_splits(synthetic_dataset_dir, seed=99)
    assert s1 != s3
