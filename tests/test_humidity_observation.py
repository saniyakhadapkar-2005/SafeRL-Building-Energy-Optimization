import numpy as np
from environment.building_env import MultiFloorBuildingEnv


def test_outdoor_humidity_is_in_policy_observation():
    env = MultiFloorBuildingEnv()
    env.reset(seed=7)
    env.outdoor_humidity = 30.0
    dry_obs = env._get_observation()
    env.outdoor_humidity = 85.0
    humid_obs = env._get_observation()
    assert dry_obs.shape == (14,)
    assert humid_obs.shape == (14,)
    assert dry_obs[4] != humid_obs[4]


def test_outdoor_humidity_changes_simulated_indoor_humidity_trend():
    dry_env = MultiFloorBuildingEnv()
    humid_env = MultiFloorBuildingEnv()
    dry_env.reset(seed=11)
    humid_env.reset(seed=11)
    dry_env.humidity = humid_env.humidity = 50.0
    dry_env.outdoor_humidity = 25.0
    humid_env.outdoor_humidity = 90.0
    actions = np.array([0, 0, 0])
    assert humid_env._calculate_humidity_change(actions) > dry_env._calculate_humidity_change(actions)
