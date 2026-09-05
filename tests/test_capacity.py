from __future__ import annotations

import pytest

from src.optimization.capacity import DEFAULT_PANEL_POWER_W, estimate_capacity


def test_capacity_basic_calculation() -> None:
    result = estimate_capacity(num_panels=20, panel_power_w=450)
    assert result.capacity_kwp == pytest.approx(9.0)


def test_capacity_zero_panels() -> None:
    result = estimate_capacity(num_panels=0, panel_power_w=450)
    assert result.capacity_kwp == 0.0


def test_capacity_uses_default_panel_power() -> None:
    result = estimate_capacity(num_panels=10)
    assert result.panel_power_w == DEFAULT_PANEL_POWER_W
    assert result.capacity_kwp == pytest.approx(10 * DEFAULT_PANEL_POWER_W / 1000)


def test_capacity_rejects_negative_panels() -> None:
    with pytest.raises(ValueError):
        estimate_capacity(num_panels=-1)


def test_capacity_rejects_nonpositive_power() -> None:
    with pytest.raises(ValueError):
        estimate_capacity(num_panels=5, panel_power_w=0)


def test_capacity_never_flags_as_energy_yield() -> None:
    result = estimate_capacity(num_panels=5, panel_power_w=450)
    assert result.is_energy_yield_estimate is False
