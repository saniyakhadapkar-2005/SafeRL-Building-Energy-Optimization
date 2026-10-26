import json

import pytest
from stable_baselines3 import PPO

from environment.building_env import MultiFloorBuildingEnv
from rl import model_io
from rl.evaluate import evaluate


def test_missing_model(tmp_path):
    model, status = model_io.load_model(tmp_path / "nope.zip")
    assert model is None and status.state == model_io.STATE_MISSING and "train" in status.message.lower()


def test_v3_model_compatible(tiny_model_v3):
    model, status = model_io.load_model(tiny_model_v3)
    assert status.ready and not status.is_legacy
    env = MultiFloorBuildingEnv(observation_version=status.observation_version)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(action)


def test_legacy_v2_model_detected(tiny_model_v2):
    model, status = model_io.load_model(tiny_model_v2)
    assert status.ready and status.is_legacy and status.observation_version == "v2"


def test_incompatible_observation_and_action_space(tmp_path):
    import gymnasium as gym, numpy as np
    from gymnasium import spaces

    class Odd(gym.Env):
        observation_space = spaces.Box(0, 1, (7,), np.float32)
        action_space = spaces.MultiDiscrete([4, 4, 4])
        def reset(self, seed=None, options=None):
            super().reset(seed=seed); return np.zeros(7, np.float32), {}
        def step(self, a):
            return np.zeros(7, np.float32), 0.0, False, True, {}

    m = PPO("MlpPolicy", Odd(), n_steps=16, batch_size=16, device="cpu")
    m.save(str(tmp_path / "odd"))
    model, status = model_io.load_model(tmp_path / "odd.zip")
    assert model is None and status.state == model_io.STATE_INCOMPATIBLE and "7" in status.message

    class Other(Odd):
        observation_space = spaces.Box(0, 1, (14,), np.float32)
        action_space = spaces.Discrete(4)
        def reset(self, seed=None, options=None):
            return np.zeros(14, np.float32), {}
        def step(self, a):
            return np.zeros(14, np.float32), 0.0, False, True, {}
    m2 = PPO("MlpPolicy", Other(), n_steps=16, batch_size=16, device="cpu")
    m2.save(str(tmp_path / "other"))
    _, st2 = model_io.load_model(tmp_path / "other.zip")
    assert st2.state == model_io.STATE_INCOMPATIBLE and "action space" in st2.message


def test_corrupt_model_file_reports_error(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"not a zip")
    model, status = model_io.load_model(p)
    assert model is None and status.state == model_io.STATE_ERROR


def test_evaluate_writes_files(tiny_model_v3, tmp_path):
    out = tmp_path / "res"
    r = evaluate(episodes=2, steps=24, data_dir=tmp_path / "nodata", out_dir=out, model_path=tiny_model_v3)
    assert (out / "evaluation_steps.csv").exists() and (out / "evaluation_summary.csv").exists()
    assert set(r["summary"]["controller"]) == {"ppo", "baseline"}
    assert (r["summary"]["steps"] == 48).all()
    meta = json.loads((out / "evaluation_meta.json").read_text())
    assert "synthetic" in meta["scenario_source"]


def test_evaluate_without_model_raises(tmp_path):
    with pytest.raises(RuntimeError):
        evaluate(episodes=1, steps=24, out_dir=tmp_path, model_path=tmp_path / "missing.zip")


def test_analyze_handles_missing_and_empty_files(tmp_path, monkeypatch):
    from rl import analyze_results as A
    monkeypatch.setattr(A, "SOURCE", tmp_path / "evaluation_steps.csv")
    assert A.main() == 1
    (tmp_path / "evaluation_steps.csv").write_text("")
    # empty file: pandas raises EmptyDataError -> must not crash silently into zeros
    with pytest.raises(Exception):
        A.main()
