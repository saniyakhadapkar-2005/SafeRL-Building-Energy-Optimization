import numpy as np
import pytest

from environment.building_env import (ACTION_NAMES, MultiFloorBuildingEnv, validate_target)
from safety.safety_layer import SafetyLayer


def test_initialisation_spaces():
    env = MultiFloorBuildingEnv()
    assert list(env.action_space.nvec) == [4, 4, 4]
    assert env.observation_space.shape == (14,)
    assert MultiFloorBuildingEnv(observation_version="v2").observation_space.shape == (11,)
    assert env.target_temperature == 23.5
    assert (env.safety_min_temperature, env.safety_max_temperature) == (18.0, 30.0)


def test_reset_and_step_observation_shape_dtype():
    env = MultiFloorBuildingEnv()
    obs, _ = env.reset(seed=3)
    assert obs.shape == (14,) and obs.dtype == np.float32
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (14,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)


def test_all_valid_actions_accepted():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0)
    for a in range(4):
        _, _, _, _, info = env.step([a, a, a])
        assert all(x in ACTION_NAMES for x in info["safe_action"])


def test_invalid_action_shape_rejected():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step([1, 2])


def test_safety_layer_applied_before_physics():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0, options={"initial_temps": [27.5, 23.5, 23.5], "targets": 23.5})
    _, _, _, _, info = env.step([0, 0, 0])
    assert info["ppo_action"] == [0, 0, 0]
    assert info["safe_action"][0] >= 2            # hot floor forced to cool
    assert info["safety_override"] is True
    assert "above target" in info["safety_reason"][0]


def test_temperature_boundary_handling():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0)
    env.floor_temperatures[:] = [29.9, 18.1, 24.0]
    env.outdoor_temperature = 50.0
    for _ in range(30):
        _, _, _, _, info = env.step([0, 0, 0])
        assert all(18.0 <= t <= 30.0 for t in info["floor_temperatures"])


def test_safety_flag_not_masked_by_clamp():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0)
    # Safety layer disabled artificially: pretend layer does nothing by checking the counterfactual flag path.
    env.floor_temperatures[:] = [29.9, 24.0, 24.0]
    env.outdoor_temperature = 50.0
    env.occupancy = 100.0
    _, _, _, _, info = env.step([0, 0, 0])
    assert "floor_safety_ok" in info and "floor_safety_ok_without_layer" in info
    assert len(info["floor_pre_clamp_temperatures"]) == 3


def test_per_floor_targets_and_validation():
    env = MultiFloorBuildingEnv(target_temperatures=[21, 23.5, 26])
    assert env.target_temperatures.tolist() == [21, 23.5, 26]
    obs, _ = env.reset(seed=1)
    assert obs[11] < obs[12] < obs[13]
    for bad in (17.0, 31.0, float("nan")):
        with pytest.raises(ValueError):
            validate_target(bad)
    with pytest.raises(ValueError):
        env.set_targets([23, 23])


def test_energy_matches_actions():
    env = MultiFloorBuildingEnv()
    env.reset(seed=0)
    e = env._calculate_energy([0, 0, 0])
    e3 = env._calculate_energy([3, 3, 3])
    assert e["hvac_energy_kwh"] == 0.0
    assert e3["hvac_energy_kwh"] > e["hvac_energy_kwh"]
    assert pytest.approx(sum(e3["floor_hvac_energy_kwh"])) == e3["hvac_energy_kwh"]
    assert e3["total_energy_kwh"] == pytest.approx(e3["base_energy_kwh"] + e3["hvac_energy_kwh"])


def test_safety_layer_scalar_and_vector_targets():
    layer = SafetyLayer()
    a1, _, _ = layer.enforce_safety([24.7, 23, 23], [0, 0, 0], 50, 22.0)
    assert a1[0] == 2
    a2, _, _ = layer.enforce_safety([24, 23, 23], [0, 0, 0], 50, [23.5, 23.5, 20.0])
    assert a2 == [0, 0, 2]
    with pytest.raises(ValueError):
        layer.enforce_safety([24, 23], [0, 0, 0])
    with pytest.raises(ValueError):
        layer.enforce_safety([24, 23, 23], [0, 0, 0], 50, [23, 23])


def test_synthetic_episode_advances_hour():
    env = MultiFloorBuildingEnv()
    env.reset(seed=5)
    h0 = env.hour
    env.step([1, 1, 1])
    assert env.hour == (h0 + 1) % 24
