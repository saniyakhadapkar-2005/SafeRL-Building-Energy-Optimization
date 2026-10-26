"""Train the PPO agent.

    python -m rl.train                      # default budget
    python -m rl.train --timesteps 500000   # longer training

Training only happens when you run this module (or press the Train button in the
dashboard, which launches this module in a separate process). Opening or
refreshing the dashboard never trains.

The agent trains in the simulation environment with randomised start
temperatures and per-floor targets. If ``data/`` contains the CSV files, the
chronological *training* split is used; otherwise the built-in synthetic
(domain-randomised) environment is used and the saved metadata says so.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from environment.building_env import OBS_FEATURES_V3, MultiFloorBuildingEnv
from rl.model_io import MODEL_DIR, MODEL_PATH, TRAIN_STATUS_PATH, meta_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMESTEPS = 300_000
N_ENVS = 4
GAMMA = 0.9


def _write_status(**fields) -> None:
    """Atomic JSON write so the dashboard never reads a half-written file."""
    TRAIN_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if TRAIN_STATUS_PATH.exists():
        try:
            current = json.loads(TRAIN_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    current.update(fields)
    current["heartbeat"] = time.time()
    tmp = TRAIN_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current), encoding="utf-8")
    os.replace(tmp, TRAIN_STATUS_PATH)


def _make_progress_callback(total: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last = 0.0

        def _on_step(self) -> bool:
            now = time.time()
            if now - self._last >= 1.0:
                self._last = now
                _write_status(state="running", timesteps=int(self.num_timesteps), total=total)
            return True

    return ProgressCallback()


def build_training_data(data_dir: Path):
    """Return ``(train_df_or_None, description)``."""
    from utils.preprocessing import data_files_present, load_prepared_data, split_train_eval
    if not data_files_present(data_dir):
        return None, "synthetic domain-randomised environment (no CSV data found)"
    df = load_prepared_data(data_dir, sample_size=100000)
    train_df, _ = split_train_eval(df, 0.2)
    return train_df, f"dataset training split ({len(train_df):,} rows; the last 20% is held out for evaluation)"


def train(timesteps: int = DEFAULT_TIMESTEPS, seed: int = 42, data_dir: Path = ROOT / "data",
          model_path: Path = MODEL_PATH) -> Path:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    timesteps = int(timesteps)
    if timesteps < 1000:
        raise ValueError("--timesteps must be at least 1000.")
    started = datetime.now(timezone.utc).isoformat()
    _write_status(state="running", timesteps=0, total=timesteps, started_at=started, pid=os.getpid(), message="")

    train_df, data_desc = build_training_data(Path(data_dir))
    print(f"Training data: {data_desc}")

    def make_env():
        return MultiFloorBuildingEnv(data=train_df, randomize_start=True, observation_version="v3")

    check_env(make_env(), warn=True)
    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    # Reward-only normalisation stabilises learning (the raw reward is ~20 per step). Observations are
    # NOT normalised here, so the saved policy can be used directly for inference.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=512, batch_size=256, n_epochs=10,
                learning_rate=3e-4, gamma=GAMMA, ent_coef=0.01, seed=seed, device="cpu")
    model.learn(total_timesteps=timesteps, callback=_make_progress_callback(timesteps))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path.with_suffix("")))
    meta = {
        "observation_version": "v3", "features": list(OBS_FEATURES_V3), "action_space": "MultiDiscrete([4,4,4])",
        "total_timesteps": timesteps, "seed": seed, "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data": data_desc, "stable_baselines3": sb3.__version__,
        "note": "Trained in a simplified simulation; not validated on a physical building.",
    }
    meta_path(model_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_status(state="finished", timesteps=timesteps, total=timesteps, model_path=str(model_path),
                  finished_at=datetime.now(timezone.utc).isoformat())
    print(f"Saved model to: {model_path}")
    return model_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the Safe-RL building PPO agent.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    try:
        train(args.timesteps, args.seed, args.data_dir)
    except Exception as exc:
        _write_status(state="failed", message=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
