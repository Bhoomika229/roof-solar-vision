"""Shared pytest fixtures: synthetic dataset on disk for tests that need real files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.data.synthetic import SyntheticRoofGenerator


@pytest.fixture(scope="session")
def synthetic_generator() -> SyntheticRoofGenerator:
    return SyntheticRoofGenerator(image_size=128, segment_scheme=9, gsd_m_per_px=0.10, seed=123)


@pytest.fixture(scope="session")
def synthetic_dataset_dir(tmp_path_factory, synthetic_generator: SyntheticRoofGenerator) -> Path:
    import json

    import cv2

    root = tmp_path_factory.mktemp("synthetic_ds")
    (root / "images").mkdir()
    (root / "masks_segments").mkdir()
    (root / "masks_superstructures").mkdir()
    (root / "metadata").mkdir()

    for i in range(10):
        sample_id = f"sample_{i:03d}"
        sample = synthetic_generator.generate(sample_id)
        cv2.imwrite(str(root / "images" / f"{sample_id}.png"), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks_segments" / f"{sample_id}.png"), sample.segment_mask)
        cv2.imwrite(str(root / "masks_superstructures" / f"{sample_id}.png"), sample.superstructure_mask)
        (root / "metadata" / f"{sample_id}.json").write_text(json.dumps(sample.metadata), encoding="utf-8")

    return root
