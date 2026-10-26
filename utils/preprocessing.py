"""Load and prepare building energy / weather / metadata CSV files.

Expected files in ``data/`` (ASHRAE-style layout, as in the original project):
``train.csv`` (building_id, meter, timestamp, meter_reading),
``weather_train.csv`` (site_id, timestamp, air_temperature, dew_temperature, ...),
``building_metadata.csv`` (site_id, building_id, square_feet, ...).

Fixes versus the original version
---------------------------------
* The old code sampled random rows of *all* buildings, so the environment
  stepped through shuffled, mixed-building data. We now select one building and
  one meter and keep rows in chronological order (one row = one hour).
* ``square_feet`` was never converted to the ``building_area_m2`` column the
  environment reads, so the real area was ignored. It is converted now.
* The dataset has no relative humidity, so the environment's humidity input was
  silently a default. It is now derived from air and dew-point temperature
  (Magnus formula) when both exist.
* Added a chronological train/evaluation split so training and evaluation do
  not share rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SQFT_TO_M2 = 0.09290304
REQUIRED_FILES = ("train.csv", "weather_train.csv", "building_metadata.csv")


def data_files_present(data_dir=DATA_DIR) -> bool:
    return all((Path(data_dir) / f).exists() for f in REQUIRED_FILES)


def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    paths = [data_dir / f for f in REQUIRED_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV file(s): " + ", ".join(missing))
    return pd.read_csv(paths[0]), pd.read_csv(paths[1]), pd.read_csv(paths[2])


def relative_humidity_from_dew_point(air_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """Magnus approximation of relative humidity (%) from air and dew-point temperature (°C)."""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * dew_c / (b + dew_c)) / np.exp(a * air_c / (b + air_c))
    return rh.clip(0.0, 100.0)


def _select_building(building: pd.DataFrame, building_id: Optional[int], meter: Optional[int]) -> pd.DataFrame:
    if "meter" in building.columns:
        if meter is None:
            meter = int(building["meter"].mode().iloc[0]) if len(building) else 0
        building = building[building["meter"] == meter]
    if "building_id" in building.columns and len(building):
        if building_id is None:
            building_id = int(building["building_id"].value_counts().idxmax())
        building = building[building["building_id"] == building_id]
    return building


def prepare_data(building, weather, metadata, sample_size=None, random_state=42,
                 building_id: Optional[int] = None, meter: Optional[int] = None) -> pd.DataFrame:
    """Return a chronological hourly table ready for ``MultiFloorBuildingEnv(data=...)``.

    ``random_state`` is accepted for backwards compatibility but unused: rows are
    no longer sampled randomly.
    """
    building = building.copy()
    weather = weather.copy()
    metadata = metadata.copy()
    for df in (building, weather):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "timestamp" in building.columns:
        building = building.dropna(subset=["timestamp"])
    building = _select_building(building, building_id, meter)
    if "timestamp" in building.columns:
        building = building.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if sample_size and len(building) > sample_size:
        building = building.head(int(sample_size))  # contiguous, chronological

    if "building_id" in building and "building_id" in metadata:
        building = building.merge(metadata, on="building_id", how="left", suffixes=("", "_meta"))
    join_keys = [k for k in ("site_id", "timestamp") if k in building.columns and k in weather.columns]
    if join_keys:
        building = building.merge(weather, on=join_keys, how="left", suffixes=("", "_weather"))
    building = building.sort_values("timestamp").reset_index(drop=True) if "timestamp" in building else building.reset_index(drop=True)

    if "building_area_m2" not in building and "square_feet" in building:
        building["building_area_m2"] = pd.to_numeric(building["square_feet"], errors="coerce") * SQFT_TO_M2
    if ("relative_humidity_2m" not in building and "air_temperature" in building
            and "dew_temperature" in building):
        building["relative_humidity_2m"] = relative_humidity_from_dew_point(
            pd.to_numeric(building["air_temperature"], errors="coerce"),
            pd.to_numeric(building["dew_temperature"], errors="coerce"))

    # Short gaps in weather-like columns are interpolated in time order; nothing is zero-filled.
    for col in ("air_temperature", "dew_temperature", "relative_humidity_2m", "building_area_m2"):
        if col in building:
            building[col] = pd.to_numeric(building[col], errors="coerce").interpolate(limit=6, limit_direction="both")

    if "timestamp" in building.columns:
        ts = pd.to_datetime(building["timestamp"], errors="coerce")
        building["hour"] = ts.dt.hour.fillna(12).astype(int)
        building["day"] = ts.dt.day.fillna(1).astype(int)
        building["month"] = ts.dt.month.fillna(1).astype(int)
        building["weekday"] = ts.dt.day_of_week.fillna(0).astype(int)
        building["is_weekend"] = (building["weekday"] >= 5).astype(int)
    else:
        for c, v in (("hour", 12), ("day", 1), ("month", 1), ("weekday", 0), ("is_weekend", 0)):
            building[c] = v
    if "occupancy" not in building:
        building["occupancy"] = building["hour"].map(
            lambda h: 5 if h < 7 or h >= 20 else (80 if h < 9 or h >= 17 else 60))
    if "electricity_price" not in building:
        building["electricity_price"] = building["hour"].map(
            lambda h: 4 if h < 7 else (7 if h < 10 else (12 if h < 17 else (9 if h < 22 else 5))))
    if "solar_generation" not in building:
        building["solar_generation"] = 0.0
    return building.reset_index(drop=True)


def split_train_eval(df: pd.DataFrame, eval_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split so evaluation rows are strictly later than training rows."""
    if not (0.0 < eval_fraction < 1.0):
        raise ValueError("eval_fraction must be between 0 and 1.")
    cut = int(len(df) * (1.0 - eval_fraction))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def load_prepared_data(data_dir=DATA_DIR, sample_size=100000) -> pd.DataFrame:
    building, weather, metadata = load_data(data_dir)
    return prepare_data(building, weather, metadata, sample_size=sample_size)
