"""Installed solar capacity from an actual panel placement (not from raw area).

    capacity_kWp = number_of_panels * panel_power_w / 1000

This is a theoretical *installed nameplate capacity* figure derived from geometrically placed
panels. It is explicitly NOT an annual energy-yield simulation (that would additionally require
irradiance data, azimuth/tilt-dependent performance ratios, temperature derating, inverter
losses, and shading time-series analysis — out of scope, and documented as future work in
``docs/research_report.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PANEL_POWER_W = 450.0  # a reasonable, common contemporary residential module rating


@dataclass
class CapacityEstimate:
    num_panels: int
    panel_power_w: float
    capacity_kwp: float
    is_energy_yield_estimate: bool = False  # always False: nameplate capacity only


def estimate_capacity(num_panels: int, panel_power_w: float = DEFAULT_PANEL_POWER_W) -> CapacityEstimate:
    if num_panels < 0:
        raise ValueError("num_panels must be non-negative")
    if panel_power_w <= 0:
        raise ValueError("panel_power_w must be positive")
    capacity_kwp = num_panels * panel_power_w / 1000.0
    return CapacityEstimate(num_panels=num_panels, panel_power_w=panel_power_w, capacity_kwp=capacity_kwp)
