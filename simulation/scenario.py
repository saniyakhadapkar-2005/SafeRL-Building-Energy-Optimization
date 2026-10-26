"""Build the exogenous per-step inputs for a simulation from a weather source.

One simulation step = one hour. The returned DataFrame is passed straight to
``MultiFloorBuildingEnv(data=...)`` so every environment step consumes one row.

Columns: timestamp, hour, air_temperature, relative_humidity_2m, occupancy,
electricity_price, solar_generation, building_area_m2, weather_source,
weather_status.

Occupancy, tariff and PV size are user-configured *assumptions*, never measured.
Weather columns come from the selected weather snapshot (live / stale /
simulated) or from a historical CSV and are labelled accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from environment.building_env import occupancy_fraction
from utils.weather import WeatherSnapshot

PV_PERFORMANCE_RATIO = 0.75
MIN_STEPS, MAX_STEPS = 6, 168


@dataclass(frozen=True)
class ScenarioConfig:
    steps: int = 24
    occupancy_peak: float = 40.0        # 0-100 scale used by the environment
    tariff_inr_per_kwh: float = 8.0     # user-configured flat tariff (not live pricing)
    pv_capacity_kw: float = 0.0         # 0 = no on-site solar
    building_area_m2: float = 2500.0

    def validate(self) -> "ScenarioConfig":
        if not (MIN_STEPS <= int(self.steps) <= MAX_STEPS):
            raise ValueError(f"Simulation duration must be {MIN_STEPS}-{MAX_STEPS} hourly steps.")
        if not (0.0 <= self.occupancy_peak <= 100.0):
            raise ValueError("Occupancy must be between 0 and 100.")
        if not (0.0 <= self.tariff_inr_per_kwh <= 50.0):
            raise ValueError("Electricity tariff must be between 0 and 50 ₹/kWh.")
        if not (0.0 <= self.pv_capacity_kw <= 500.0):
            raise ValueError("PV capacity must be between 0 and 500 kW.")
        if not (100.0 <= self.building_area_m2 <= 100000.0):
            raise ValueError("Building area must be between 100 and 100000 m².")
        return self


class ScenarioError(Exception):
    pass


def _assemble(times: pd.Series, temp: np.ndarray, hum: np.ndarray, radiation: np.ndarray,
              cfg: ScenarioConfig, source: str, status: str) -> pd.DataFrame:
    hours = pd.to_datetime(times).dt.hour.to_numpy()
    occupancy = np.array([cfg.occupancy_peak * occupancy_fraction(h) for h in hours])
    solar = cfg.pv_capacity_kw * PV_PERFORMANCE_RATIO * np.nan_to_num(radiation, nan=0.0) / 1000.0
    return pd.DataFrame({
        "timestamp": pd.to_datetime(times).reset_index(drop=True),
        "hour": hours.astype(float),
        "air_temperature": temp, "relative_humidity_2m": hum,
        "occupancy": occupancy, "electricity_price": cfg.tariff_inr_per_kwh,
        "solar_generation": solar, "building_area_m2": cfg.building_area_m2,
        "weather_source": source, "weather_status": status,
    })


def _fill(series: pd.Series, name: str, max_gap: int = 3) -> np.ndarray:
    """Interpolate short gaps only; refuse to invent long stretches of data."""
    s = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    s = s.interpolate(limit=max_gap, limit_direction="both")
    if s.isna().any():
        raise ScenarioError(f"Weather data has too many missing '{name}' values to run a simulation.")
    return s.to_numpy(dtype=float)


def from_weather_snapshot(snapshot: WeatherSnapshot, cfg: ScenarioConfig,
                          start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Hourly scenario starting at the current local hour of the snapshot."""
    cfg.validate()
    h = snapshot.hourly
    if h is None or h.empty or "outdoor_temperature_c" not in h.columns:
        raise ScenarioError("This weather source has no hourly series to drive a simulation.")
    h = h.sort_values("timestamp").reset_index(drop=True)
    if start is None:
        cur = snapshot.current.get("time")
        start = pd.to_datetime(cur, errors="coerce") if cur else pd.NaT
        if pd.isna(start):
            start = h["timestamp"].iloc[0]
    start = pd.Timestamp(start).floor("h")
    window = h[h["timestamp"] >= start].head(cfg.steps).reset_index(drop=True)
    if len(window) < cfg.steps:
        raise ScenarioError(
            f"Only {len(window)} hourly weather values are available from {start:%Y-%m-%d %H:%M}; "
            f"{cfg.steps} are needed. Shorten the duration or refresh the weather.")
    temp = _fill(window["outdoor_temperature_c"], "temperature")
    if window["outdoor_humidity_pct"].isna().all():
        raise ScenarioError("Weather data has no humidity values.")
    hum = _fill(window["outdoor_humidity_pct"], "humidity")
    rad = pd.to_numeric(window.get("shortwave_radiation_wm2", pd.Series(np.nan, index=window.index)),
                        errors="coerce").to_numpy(dtype=float)
    return _assemble(window["timestamp"], temp, hum, rad, cfg, snapshot.source, snapshot.status)


def from_historical_frame(df: pd.DataFrame, cfg: ScenarioConfig, offset: int = 0) -> pd.DataFrame:
    """Scenario from a prepared historical CSV window (``utils.preprocessing``)."""
    cfg.validate()
    needed = {"timestamp", "air_temperature"}
    if df is None or df.empty or not needed.issubset(df.columns):
        raise ScenarioError("Historical dataset is missing or lacks 'timestamp'/'air_temperature' columns.")
    window = df.iloc[offset: offset + cfg.steps].reset_index(drop=True)
    if len(window) < cfg.steps:
        raise ScenarioError(f"Historical dataset has only {len(window)} rows from offset {offset}.")
    temp = _fill(window["air_temperature"], "air_temperature")
    hum_col = "relative_humidity_2m" if "relative_humidity_2m" in window else None
    hum = _fill(window[hum_col], "humidity") if hum_col else np.full(len(window), 50.0)
    out = _assemble(window["timestamp"], temp, hum, np.full(len(window), np.nan), cfg,
                    "Historical CSV dataset (not live)", "historical")
    if hum_col is None:
        out["weather_status"] = "historical (humidity missing - constant 50% assumed)"
    return out


def synthetic_scenario(seed: int, steps: int = 168, start: str = "2024-01-01") -> pd.DataFrame:
    """Reproducible randomised scenario for evaluation when no dataset is available.

    Same functional form as the environment's synthetic training episodes
    (diurnal outdoor temperature/humidity, office occupancy profile, daylight
    solar). Labelled as simulated; it is NOT weather data.
    """
    rng = np.random.default_rng(seed)
    base_t, amp = rng.uniform(18.0, 38.0), rng.uniform(2.0, 6.0)
    base_h, occ_peak = rng.uniform(30.0, 85.0), rng.uniform(20.0, 100.0)
    solar_peak, price = rng.uniform(0.0, 80.0), rng.uniform(4.0, 16.0)
    start_hour = int(rng.integers(0, 24))
    times = pd.date_range(pd.Timestamp(start) + pd.Timedelta(hours=start_hour), periods=steps, freq="h")
    hours = times.hour.to_numpy(dtype=float)
    phase = np.sin(2 * np.pi * (hours - 9.0) / 24.0)
    return pd.DataFrame({
        "timestamp": times, "hour": hours,
        "air_temperature": np.clip(base_t + amp * phase, 0.0, 50.0),
        "relative_humidity_2m": np.clip(base_h - 2.0 * amp * phase, 0.0, 100.0),
        "occupancy": [occ_peak * occupancy_fraction(h) for h in hours],
        "electricity_price": price,
        "solar_generation": solar_peak * np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0)),
        "building_area_m2": 2500.0,
        "weather_source": "Synthetic evaluation scenario (simulated, not weather)",
        "weather_status": "simulated",
    })
