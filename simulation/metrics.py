"""Pure metric functions over simulation / evaluation step tables.

Every percentage is returned together with its numerator and denominator so the
UI can state exactly what was counted. Missing data yields ``None`` - never 0.

Units: ``*_kwh`` columns are kWh per simulation step (one step = one hour).
Floor-level columns use the pattern ``floor{1,2,3}_<name>``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from environment.building_env import COMFORT_MAX_C, COMFORT_MIN_C, NUM_FLOORS, SAFETY_MAX_C, SAFETY_MIN_C

FLOORS = range(1, NUM_FLOORS + 1)


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int

    @property
    def pct(self) -> Optional[float]:
        return None if self.denominator == 0 else 100.0 * self.numerator / self.denominator

    def describe(self, unit: str = "floor-steps") -> str:
        if self.denominator == 0:
            return "not available (no evaluated steps)"
        return f"{self.pct:.1f}% ({self.numerator} of {self.denominator} {unit})"


def _floor_cols(df: pd.DataFrame, name: str, floors=None) -> list[str]:
    floors = list(floors) if floors is not None else list(FLOORS)
    return [f"floor{i}_{name}" for i in floors if f"floor{i}_{name}" in df.columns]


def comfort_ratio(df: pd.DataFrame, floors=None) -> Ratio:
    """Floor-steps whose temperature is inside 22-25 °C / all floor-steps evaluated."""
    cols = _floor_cols(df, "temp", floors)
    if df is None or df.empty or not cols:
        return Ratio(0, 0)
    t = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = t.notna()
    inside = (t >= COMFORT_MIN_C) & (t <= COMFORT_MAX_C) & valid
    return Ratio(int(inside.to_numpy().sum()), int(valid.to_numpy().sum()))


def safety_ratio(df: pd.DataFrame, floors=None, without_layer: bool = False) -> Ratio:
    """Floor-steps within 18-30 °C.

    ``without_layer=False``: measured on the physical result of the *safe*
    (post-Safety-Layer) action, before the emergency clamp.
    ``without_layer=True``: one-step counterfactual - the raw agent action
    applied to the same state with no Safety Layer.
    """
    name = "cf_temp" if without_layer else "preclamp_temp"
    cols = _floor_cols(df, name, floors)
    if df is None or df.empty or not cols:
        return Ratio(0, 0)
    t = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = t.notna()
    inside = (t >= SAFETY_MIN_C) & (t <= SAFETY_MAX_C) & valid
    return Ratio(int(inside.to_numpy().sum()), int(valid.to_numpy().sum()))


def override_ratio(df: pd.DataFrame) -> Ratio:
    if df is None or df.empty or "safety_override" not in df.columns:
        return Ratio(0, 0)
    flags = df["safety_override"].map(_to_bool)
    return Ratio(int(flags.sum()), int(flags.notna().sum()))


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _sum(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return None if s.notna().sum() == 0 else float(s.sum())


def _mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return None if s.notna().sum() == 0 else float(s.mean())


def floor_energy_kwh(df: pd.DataFrame, floor: int) -> Optional[float]:
    return _sum(df, f"floor{floor}_hvac_kwh")


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate a step table into a metrics dictionary (values may be ``None``)."""
    n = 0 if df is None else len(df)
    comfort, safety = comfort_ratio(df), safety_ratio(df)
    safety_cf, overrides = safety_ratio(df, without_layer=True), override_ratio(df)
    return {
        "steps": n,
        "total_reward": _sum(df, "reward"),
        "mean_reward": _mean(df, "reward"),
        "total_energy_kwh": _sum(df, "total_energy_kwh"),
        "hvac_energy_kwh": _sum(df, "hvac_energy_kwh"),
        "grid_energy_kwh": _sum(df, "grid_energy_kwh"),
        "total_cost_inr": _sum(df, "electricity_cost_inr"),
        "mean_indoor_temp_c": _mean(df, "indoor_temperature"),
        "comfort_pct": comfort.pct, "comfort_num": comfort.numerator, "comfort_den": comfort.denominator,
        "safety_pct": safety.pct, "safety_num": safety.numerator, "safety_den": safety.denominator,
        "safety_without_layer_pct": safety_cf.pct,
        "safety_without_layer_num": safety_cf.numerator, "safety_without_layer_den": safety_cf.denominator,
        "override_count": overrides.numerator, "override_den": overrides.denominator,
        "override_pct": overrides.pct,
    }


def fmt(value: Optional[float], spec: str = ".1f", unit: str = "", missing: str = "n/a") -> str:
    """Format a metric, showing ``missing`` (not 0) for ``None``/NaN."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return missing
    return f"{value:{spec}}{unit}"
