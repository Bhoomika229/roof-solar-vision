from __future__ import annotations

import numpy as np

from src.data.synthetic import SyntheticRoofGenerator
from src.pipeline import SolarConfig, run_pipeline


def test_synthetic_generator_produces_consistent_shapes() -> None:
    gen = SyntheticRoofGenerator(image_size=128, segment_scheme=9, seed=1)
    sample = gen.generate("s1")
    assert sample.image.shape == (128, 128, 3)
    assert sample.segment_mask.shape == (128, 128)
    assert sample.superstructure_mask.shape == (128, 128)
    assert sample.image.dtype == np.uint8


def test_synthetic_generator_is_seeded_reproducibly() -> None:
    gen1 = SyntheticRoofGenerator(image_size=64, seed=42)
    gen2 = SyntheticRoofGenerator(image_size=64, seed=42)
    s1 = gen1.generate("a")
    s2 = gen2.generate("a")
    assert np.array_equal(s1.segment_mask, s2.segment_mask)
    assert np.array_equal(s1.image, s2.image)


def test_synthetic_metadata_documents_it_is_not_a_real_measurement() -> None:
    gen = SyntheticRoofGenerator(image_size=64, seed=3)
    sample = gen.generate("s")
    assert "not derived from the rendered pixels" in sample.metadata["note"]


def test_pipeline_runs_end_to_end_on_synthetic_masks() -> None:
    gen = SyntheticRoofGenerator(image_size=128, segment_scheme=9, gsd_m_per_px=0.1, seed=5)
    sample = gen.generate("s")
    result = run_pipeline(
        sample.image,
        solar_config=SolarConfig(segment_scheme=9),
        gsd_m_per_px=0.1,
        segment_mask=sample.segment_mask,
        superstructure_mask=sample.superstructure_mask,
    )
    assert result.usable.roof_area.area_m2 >= result.usable.usable_area.area_m2
    assert result.capacity.capacity_kwp >= 0
    assert result.uncertainty.panel_count_low <= result.uncertainty.panel_count_central <= result.uncertainty.panel_count_high
    lines = result.report_lines()
    assert any("Roof area" in line for line in lines)
    assert any("NOT a measurement" in line for line in lines)


def test_pipeline_with_no_gsd_flags_assumed() -> None:
    gen = SyntheticRoofGenerator(image_size=64, seed=8)
    sample = gen.generate("s")
    result = run_pipeline(
        sample.image, segment_mask=sample.segment_mask, superstructure_mask=sample.superstructure_mask,
    )
    assert result.gsd_is_assumed is True
