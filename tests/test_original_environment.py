from environment.building_env import MultiFloorBuildingEnv


def test_env_step():
    env = MultiFloorBuildingEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, details = env.step([0, 1, 2])
    assert env.observation_space.contains(obs)
    assert len(details["floor_temperatures"]) == 3
    assert len(details["safe_action"]) == 3
    assert isinstance(reward, float)
