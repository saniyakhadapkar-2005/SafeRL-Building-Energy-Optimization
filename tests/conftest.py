import numpy as np
import pandas as pd
import pytest

from environment.building_env import MultiFloorBuildingEnv


@pytest.fixture(scope="session")
def tiny_model_v3(tmp_path_factory):
    from stable_baselines3 import PPO
    env = MultiFloorBuildingEnv(randomize_start=True)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu", seed=0)
    model.learn(128)
    path = tmp_path_factory.mktemp("models") / "ppo_safe_building_v3.zip"
    model.save(str(path.with_suffix("")))
    return path


@pytest.fixture(scope="session")
def tiny_model_v2(tmp_path_factory):
    from stable_baselines3 import PPO
    env = MultiFloorBuildingEnv(observation_version="v2")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu", seed=0)
    model.learn(128)
    path = tmp_path_factory.mktemp("models2") / "legacy_v2.zip"
    model.save(str(path.with_suffix("")))
    return path


@pytest.fixture
def scenario():
    from simulation.scenario import synthetic_scenario
    return synthetic_scenario(seed=1, steps=24)
