import numpy as np
import pandas as pd
import pytest

from utils import preprocessing as P


def fixture():
    ts = pd.date_range("2016-01-01", periods=300, freq="h")
    rows = []
    for b in (0, 1):
        for i, t in enumerate(ts[: 300 if b == 0 else 100]):
            rows.append({"building_id": b, "meter": 0, "timestamp": t, "meter_reading": 10 + i})
    building = pd.DataFrame(rows).sample(frac=1, random_state=1)   # shuffled on purpose
    weather = pd.DataFrame({"site_id": 0, "timestamp": ts, "air_temperature": np.linspace(20, 30, 300), "dew_temperature": 15.0})
    meta = pd.DataFrame({"site_id": 0, "building_id": [0, 1], "square_feet": [10000, 5000]})
    return building, weather, meta


def test_chronological_single_building_with_area_and_humidity():
    b, w, m = fixture()
    df = P.prepare_data(b, w, m)
    assert df["building_id"].nunique() == 1 and df["building_id"].iloc[0] == 0   # largest building
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    assert df["building_area_m2"].iloc[0] == pytest.approx(10000 * P.SQFT_TO_M2)
    assert df["relative_humidity_2m"].between(0, 100).all()
    assert {"hour", "occupancy", "electricity_price", "solar_generation"} <= set(df.columns)


def test_sample_size_is_contiguous_head():
    b, w, m = fixture()
    df = P.prepare_data(b, w, m, sample_size=50)
    assert len(df) == 50 and df["timestamp"].is_monotonic_increasing


def test_split_has_no_overlap():
    b, w, m = fixture()
    df = P.prepare_data(b, w, m)
    tr, ev = P.split_train_eval(df, 0.2)
    assert tr["timestamp"].max() < ev["timestamp"].min() and len(tr) + len(ev) == len(df)
    with pytest.raises(ValueError):
        P.split_train_eval(df, 1.5)


def test_missing_data_files(tmp_path):
    assert not P.data_files_present(tmp_path)
    with pytest.raises(FileNotFoundError):
        P.load_data(tmp_path)


def test_env_runs_on_prepared_dataset():
    from environment.building_env import MultiFloorBuildingEnv
    b, w, m = fixture()
    df = P.prepare_data(b, w, m)
    env = MultiFloorBuildingEnv(data=df, max_steps=50, randomize_start=True)
    env.reset(seed=0)
    n = 0
    while True:
        _, _, te, tr, info = env.step([1, 1, 1]); n += 1
        if te or tr: break
    assert n == 50 and info["building_area"] == pytest.approx(10000 * P.SQFT_TO_M2)
