import numpy as np
import pandas as pd
import pytest

from dashboard.charts import align_outdoor, empty_figure, temperature_chart, energy_timeline_chart, hvac_action_chart, comfort_bar_chart, safety_chart, reward_chart, cost_chart, outdoor_indoor_chart, comfort_timeline_chart
from simulation import metrics
from simulation.runner import (CONTROLLER_BASELINE, CONTROLLER_PPO, SimConfig, SimulationError, SimulationSession,
                               baseline_action, to_long_floor_frame)
from simulation.scenario import ScenarioConfig, ScenarioError, from_weather_snapshot, from_historical_frame
from utils.weather import simulated_weather, preset_location, WeatherSnapshot, STATUS_LIVE


def run(scenario, **kw):
    cfg = SimConfig(controller=CONTROLLER_BASELINE, **kw)
    s = SimulationSession(scenario, cfg)
    return s, s.run_to_end()


def test_all_floor_simulation_uses_real_env_steps(scenario):
    s, df = run(scenario)
    assert len(df) == 24 and s.finished and s.env.current_step == 24
    assert all(f"floor{n}_temp" in df for n in (1, 2, 3))
    assert df["floor1_controlled"].all()


def test_individual_floor_scope_preserves_other_floors(scenario):
    s, df = run(scenario, scope=1)
    assert df["floor2_controlled"].all() and not df["floor1_controlled"].any()
    assert (df["floor1_requested"] == 0).all() and (df["floor3_requested"] == 0).all()   # not agent controlled
    assert df[["floor1_temp", "floor3_temp"]].notna().all().all()                          # still simulated
    # safety layer still protects uncontrolled floors
    assert df["floor1_action"].max() >= 1


def test_floor_temperatures_are_independent_per_floor(scenario):
    _, df = run(scenario, start_temps=(21.0, 25.0, 29.0))
    assert df["floor1_temp"].iloc[0] != df["floor3_temp"].iloc[0]


def test_reset_restores_initial_state(scenario):
    s = SimulationSession(scenario, SimConfig(controller=CONTROLLER_BASELINE))
    s.advance(5)
    assert s.current_step == 5
    first = s.to_frame().iloc[0]["floor1_temp"]
    s.reset()
    assert s.current_step == 0 and s.state == "ready" and s.to_frame().empty
    s.advance(1)
    assert s.to_frame().iloc[0]["floor1_temp"] == first     # deterministic after reset


def test_step_after_finish_raises(scenario):
    s, _ = run(scenario)
    with pytest.raises(SimulationError):
        s.step()


def test_ppo_requires_model(scenario):
    with pytest.raises(SimulationError):
        SimulationSession(scenario, SimConfig(controller=CONTROLLER_PPO), model=None)
    with pytest.raises(SimulationError):
        SimulationSession(scenario.iloc[0:0], SimConfig(controller=CONTROLLER_BASELINE))


def test_ppo_session_with_model(scenario, tiny_model_v3):
    from rl.model_io import load_model
    model, status = load_model(tiny_model_v3)
    assert status.ready and status.observation_version == "v3"
    s = SimulationSession(scenario, SimConfig(controller=CONTROLLER_PPO, observation_version="v3"), model=model)
    df = s.run_to_end()
    assert len(df) == 24
    assert set(df["floor1_action"].unique()) <= {0, 1, 2, 3}


def test_baseline_rule():
    assert baseline_action([25, 23.5, 22], [23.5] * 3).tolist() == [2, 0, 1]


def test_scenario_validation_and_weather_errors():
    snap = simulated_weather(preset_location("Pune"))
    with pytest.raises(ValueError):
        ScenarioConfig(steps=2).validate()
    with pytest.raises(ValueError):
        ScenarioConfig(tariff_inr_per_kwh=-1).validate()
    with pytest.raises(ScenarioError):
        from_weather_snapshot(snap, ScenarioConfig(steps=168))          # only 96 hours available
    empty = WeatherSnapshot(location=snap.location, status=STATUS_LIVE, source="x", fetched_at=None)
    with pytest.raises(ScenarioError):
        from_weather_snapshot(empty, ScenarioConfig(steps=24))
    sc = from_weather_snapshot(snap, ScenarioConfig(steps=24, pv_capacity_kw=0.0))
    assert sc["weather_status"].iloc[0] == "simulated"


def test_missing_weather_values_not_invented():
    snap = simulated_weather(preset_location("Pune"))
    h = snap.hourly.copy()
    h.loc[h.index[30:45], "outdoor_temperature_c"] = np.nan   # long gap
    snap.hourly = h
    with pytest.raises(ScenarioError):
        from_weather_snapshot(snap, ScenarioConfig(steps=24), start=h["timestamp"].iloc[26])


def test_historical_scenario():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=60, freq="h"), "air_temperature": 25.0})
    sc = from_historical_frame(df, ScenarioConfig(steps=24))
    assert "humidity missing" in sc["weather_status"].iloc[0]
    with pytest.raises(ScenarioError):
        from_historical_frame(df, ScenarioConfig(steps=24), offset=50)


# ---- metrics --------------------------------------------------------------
def test_comfort_and_safety_denominators(scenario):
    _, df = run(scenario)
    c = metrics.comfort_ratio(df)
    assert c.denominator == 24 * 3 and 0 <= c.numerator <= c.denominator
    assert c.pct == pytest.approx(100 * c.numerator / c.denominator)
    assert metrics.comfort_ratio(df, floors=[2]).denominator == 24
    assert metrics.safety_ratio(df).denominator == 72
    s = metrics.summarize(df)
    assert s["total_energy_kwh"] == pytest.approx(df["total_energy_kwh"].sum())
    assert s["total_energy_kwh"] == pytest.approx(df["base_energy_kwh"].sum() + df["hvac_energy_kwh"].sum())


def test_metrics_exact_values():
    df = pd.DataFrame({"floor1_temp": [23, 26, 23.5, 21], "reward": [1.0, 2.0, 3.0, 4.0],
                       "safety_override": ["True", "False", "false", "1"]})
    assert metrics.comfort_ratio(df, [1]).numerator == 2
    assert metrics.override_ratio(df).numerator == 2 and metrics.override_ratio(df).denominator == 4


def test_empty_and_missing_data_give_none_not_zero():
    s = metrics.summarize(pd.DataFrame())
    assert s["steps"] == 0 and s["total_energy_kwh"] is None and s["comfort_pct"] is None
    assert metrics.fmt(None) == "n/a" and metrics.fmt(float("nan")) == "n/a" and metrics.fmt(0.0) == "0.0"
    assert metrics.summarize(pd.DataFrame({"x": [1]}))["total_cost_inr"] is None
    assert "not available" in metrics.Ratio(0, 0).describe()


# ---- charts ---------------------------------------------------------------
def test_charts_build_from_real_run_and_empty(scenario):
    _, df = run(scenario)
    for f in (temperature_chart, outdoor_indoor_chart, energy_timeline_chart, hvac_action_chart,
              comfort_timeline_chart, comfort_bar_chart, safety_chart):
        assert len(f(df, [1, 3]).data) > 0
        assert f(pd.DataFrame()).layout.annotations          # empty state message
    assert len(reward_chart(df).data) == 2 and len(cost_chart(df, "x").data) == 2
    assert len(temperature_chart(df, [2]).data) == 2          # one floor: temperature + target
    assert not to_long_floor_frame(pd.DataFrame()).shape[0]


def test_align_outdoor_handles_different_intervals():
    sim = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=6, freq="h")})
    out = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="3h"), "t": [20.0, 23.0, 26.0]})
    m = align_outdoor(sim, out, "t", tolerance="90min")
    assert m["t"].notna().all()
    far = pd.DataFrame({"timestamp": [pd.Timestamp("2026-02-01")], "t": [1.0]})
    assert align_outdoor(sim, far, "t")["t"].isna().all()      # no invented values outside tolerance
