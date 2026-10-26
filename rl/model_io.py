"""Locate, load and validate trained PPO models (inference side only).

Training lives in ``rl/train.py``; evaluation in ``rl/evaluate.py``. This module
never trains anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from environment.building_env import OBS_FEATURES_V2, OBS_FEATURES_V3

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "trained_model"
RESULTS_DIR = ROOT / "outputs" / "results"
MODEL_NAME = "ppo_safe_building_v3"            # 14-feature observation (current)
LEGACY_MODEL_NAME = "ppo_safe_building_humidity_v2"  # 11-feature observation (original project)
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.zip"
LEGACY_MODEL_PATH = MODEL_DIR / f"{LEGACY_MODEL_NAME}.zip"
TRAIN_STATUS_PATH = MODEL_DIR / "training_status.json"

OBS_DIM_TO_VERSION = {len(OBS_FEATURES_V3): "v3", len(OBS_FEATURES_V2): "v2"}

STATE_READY = "ready"
STATE_MISSING = "missing"
STATE_INCOMPATIBLE = "incompatible"
STATE_ERROR = "error"


@dataclass
class ModelStatus:
    state: str
    message: str
    path: Optional[Path] = None
    observation_version: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    @property
    def is_legacy(self) -> bool:
        return self.observation_version == "v2"


def meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def read_meta(model_path: Path) -> dict:
    p = meta_path(model_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def candidate_paths(preferred: Optional[Path] = None) -> list[Path]:
    paths = [Path(preferred)] if preferred else [MODEL_PATH, LEGACY_MODEL_PATH]
    return paths


def load_model(path: Optional[Path] = None):
    """Return ``(model_or_None, ModelStatus)``. Never raises for expected problems."""
    paths = candidate_paths(path)
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None, ModelStatus(
            STATE_MISSING,
            "No trained PPO model found. Train one from the PPO Model page or run: python -m rl.train",
            path=paths[0])
    problems = []
    for p in existing:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(p), device="cpu")
        except Exception as exc:
            problems.append(ModelStatus(STATE_ERROR, f"Could not load {p.name}: {type(exc).__name__}: {exc}", path=p))
            continue
        status = check_compatibility(model, p)
        if status.ready:
            return model, status
        problems.append(status)
    return None, problems[0]


def check_compatibility(model, path: Path) -> ModelStatus:
    from gymnasium import spaces
    meta = read_meta(path)
    obs_space, act_space = model.observation_space, model.action_space
    if not isinstance(act_space, spaces.MultiDiscrete) or list(np.asarray(act_space.nvec)) != [4, 4, 4]:
        return ModelStatus(STATE_INCOMPATIBLE,
                           f"{path.name}: action space {act_space} does not match MultiDiscrete([4, 4, 4]). Retrain required.",
                           path=path, meta=meta)
    shape = getattr(obs_space, "shape", None)
    dim = shape[0] if shape and len(shape) == 1 else None
    version = OBS_DIM_TO_VERSION.get(dim)
    if version is None:
        return ModelStatus(STATE_INCOMPATIBLE,
                           f"{path.name}: expects {dim} observation features; this project provides 14 (v3) or 11 (v2). Retrain required.",
                           path=path, meta=meta)
    note = ("Current model (14 features, includes per-floor targets)." if version == "v3" else
            "Legacy v2 model (11 features): it cannot see per-floor target temperatures; "
            "targets still affect the physics and Safety Layer. Retrain for a target-aware agent.")
    return ModelStatus(STATE_READY, note, path=path, observation_version=version, meta=meta)


def read_training_status() -> dict:
    if TRAIN_STATUS_PATH.exists():
        try:
            return json.loads(TRAIN_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}
