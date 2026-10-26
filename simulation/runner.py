"""Step-based simulation session wrapping the real environment.

Every call to :meth:`SimulationSession.step` advances the actual
``MultiFloorBuildingEnv`` by one hourly step. Nothing is faked: all recorded
values come from the environment's ``info`` dictionary.

Controllers
-----------
``ppo``       the trained PPO model (deterministic inference), joint output for 3 floors
``baseline``  a transparent rule-based thermostat (clearly labelled, NOT the RL agent)

Scope
-----
``scope=None``  all three floors are controlled by the chosen controller.
``scope=k``     only floor ``k`` (0-based) takes the controller's action. The other
                floors are *not agent-controlled*: they request OFF each step and the
                Safety Layer still protects them (it may raise the action). The PPO
                model still produces one joint 3-floor output; only the selected
                floor's component is used - the other components are discarded, not
                relabelled as separate model predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from environment.building_env import ACTION_NAMES, NUM_FLOORS, MultiFloorBuildingEnv

CONTROLLER_PPO = "ppo"
CONTROLLER_BASELINE = "baseline"
CONTROLLER_LABELS = {
    CONTROLLER_PPO: "Trained PPO agent + Safety Layer",
    CONTROLLER_BASELINE: "Rule-based thermostat baseline (not RL) + Safety Layer",
}

STATE_READY, STATE_RUNNING, STATE_PAUSED, STATE_FINISHED = "ready", "running", "paused", "finished"


class SimulationError(Exception):
    pass


@dataclass(frozen=True)
class SimConfig:
    targets: tuple[float, float, float] = (23.5, 23.5, 23.5)
    start_temps: tuple[float, float, float] = (27.0, 27.0, 27.0)
    controller: str = CONTROLLER_PPO
    scope: Optional[int] = None          # None = all floors, else 0-based floor index
    seed: int = 42
    observation_version: str = "v3"


def baseline_action(temps: Sequence[float], targets: Sequence[float]) -> np.ndarray:
    """Thermostat rule: MEDIUM when >0.5°C above target, LOW when >0.5°C below, else OFF."""
    t, g = np.asarray(temps, dtype=float), np.asarray(targets, dtype=float)
    return np.where(t > g + 0.5, 2, np.where(t < g - 0.5, 1, 0)).astype(np.int64)


class SimulationSession:
    def __init__(self, scenario: pd.DataFrame, config: SimConfig, model=None):
        if scenario is None or len(scenario) == 0:
            raise SimulationError("The scenario is empty; nothing to simulate.")
        if config.controller not in CONTROLLER_LABELS:
            raise SimulationError(f"Unknown controller '{config.controller}'.")
        if config.controller == CONTROLLER_PPO and (model is None or not hasattr(model, "predict")):
            raise SimulationError("A compatible trained PPO model is required for the PPO controller.")
        if config.scope is not None and config.scope not in range(NUM_FLOORS):
            raise SimulationError("Floor scope must be None or 0, 1, 2.")
        self.scenario = scenario.reset_index(drop=True)
        self.config = config
        self.model = model
        self.env = MultiFloorBuildingEnv(
            data=self.scenario, max_steps=len(self.scenario),
            observation_version=config.observation_version,
        )
        self.state = STATE_READY
        self.records: list[dict] = []
        self._obs: Optional[np.ndarray] = None
        self.last_error: Optional[str] = None
        self.reset()

    # ------------------------------------------------------------------
    @property
    def total_steps(self) -> int:
        return len(self.scenario)

    @property
    def current_step(self) -> int:
        return len(self.records)

    @property
    def finished(self) -> bool:
        return self.state == STATE_FINISHED

    def reset(self) -> None:
        """Return to step 0 with the configured targets/start temperatures (same seed)."""
        self._obs, _ = self.env.reset(
            seed=self.config.seed,
            options={"targets": list(self.config.targets), "initial_temps": list(self.config.start_temps)},
        )
        self.records = []
        self.state = STATE_READY
        self.last_error = None

    def start(self) -> None:
        if self.state in (STATE_READY, STATE_PAUSED):
            self.state = STATE_RUNNING

    def pause(self) -> None:
        if self.state == STATE_RUNNING:
            self.state = STATE_PAUSED

    # ------------------------------------------------------------------
    def _requested_actions(self) -> np.ndarray:
        cfg = self.config
        if cfg.controller == CONTROLLER_PPO:
            action, _ = self.model.predict(self._obs, deterministic=True)
            action = np.asarray(action, dtype=np.int64).reshape(-1)
            if action.shape != (NUM_FLOORS,) or action.min() < 0 or action.max() > 3:
                raise SimulationError(f"Model returned an invalid action {action.tolist()}; expected 3 values in 0..3.")
        else:
            action = baseline_action(self.env.floor_temperatures, self.env.target_temperatures)
        if cfg.scope is None:
            return action
        masked = np.zeros(NUM_FLOORS, dtype=np.int64)
        masked[cfg.scope] = action[cfg.scope]
        return masked

    def step(self) -> dict:
        if self.state == STATE_FINISHED:
            raise SimulationError("Simulation already finished; reset it to run again.")
        try:
            requested = self._requested_actions()
            obs, reward, terminated, truncated, info = self.env.step(requested)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.state = STATE_PAUSED
            raise
        self._obs = obs
        row = self._record(info, float(reward), requested)
        self.records.append(row)
        if terminated or truncated or len(self.records) >= self.total_steps:
            self.state = STATE_FINISHED
        return row

    def advance(self, n: int = 1) -> int:
        """Advance up to ``n`` real environment steps; returns the number executed."""
        done = 0
        while done < n and self.state != STATE_FINISHED:
            self.step()
            done += 1
        return done

    def run_to_end(self, progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
        self.start()
        while self.state != STATE_FINISHED:
            self.step()
            if progress:
                progress(self.current_step, self.total_steps)
        return self.to_frame()

    # ------------------------------------------------------------------
    def _record(self, info: dict, reward: float, requested: np.ndarray) -> dict:
        cfg = self.config
        row: dict = {
            "step": len(self.records) + 1,
            "timestamp": pd.Timestamp(info["timestamp"]) if info.get("timestamp") not in (None, "") else pd.NaT,
            "hour": info["hour"],
            "controller": cfg.controller,
            "outdoor_temperature": info["outdoor_temperature"],
            "outdoor_humidity": info["outdoor_humidity"],
            "indoor_humidity": info["humidity"],
            "occupancy": info["occupancy"],
            "electricity_price": info["electricity_price"],
            "solar_generation": info["solar_generation"],
            "indoor_temperature": info["indoor_temperature"],
            "total_energy_kwh": info["total_energy_kwh"],
            "hvac_energy_kwh": info["hvac_energy_kwh"],
            "base_energy_kwh": info["base_energy_kwh"],
            "grid_energy_kwh": info["grid_energy_kwh"],
            "solar_used_kwh": info["solar_used_kwh"],
            "electricity_cost_inr": info["electricity_cost_inr"],
            "reward": reward,
            "comfort_score": info["comfort_score"],
            "safety_override": bool(info["safety_override"]),
            "post_action_override": bool(info["post_action_override"]),
            "safety_reason": " | ".join(r for r in info["safety_reason"] if r != "No override"),
        }
        for i in range(NUM_FLOORS):
            n = i + 1
            controlled = cfg.scope is None or cfg.scope == i
            row.update({
                f"floor{n}_temp": info["floor_temperatures"][i],
                f"floor{n}_target": info["floor_targets"][i],
                f"floor{n}_requested": int(requested[i]),
                f"floor{n}_action": int(info["safe_action"][i]),
                f"floor{n}_action_name": ACTION_NAMES[int(info["safe_action"][i])],
                f"floor{n}_controlled": bool(controlled),
                f"floor{n}_hvac_kwh": info["floor_hvac_energy_kwh"][i],
                f"floor{n}_preclamp_temp": info["floor_pre_clamp_temperatures"][i],
                f"floor{n}_cf_temp": info["floor_counterfactual_temperatures"][i],
                f"floor{n}_in_comfort": bool(info["floor_in_comfort"][i]),
                f"floor{n}_mode": info["hvac_modes"][i],
            })
        return row

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def latest(self) -> Optional[dict]:
        return self.records[-1] if self.records else None


def to_long_floor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide per-floor columns into one row per (step, floor) for plotting."""
    if df is None or df.empty:
        return pd.DataFrame()
    parts = []
    for n in range(1, NUM_FLOORS + 1):
        if f"floor{n}_temp" not in df.columns:
            continue
        part = pd.DataFrame({
            "step": df["step"], "timestamp": df["timestamp"], "floor": f"Floor {n}", "floor_number": n,
            "temperature": df[f"floor{n}_temp"], "target": df[f"floor{n}_target"],
            "action": df[f"floor{n}_action"], "action_name": df[f"floor{n}_action_name"],
            "requested": df[f"floor{n}_requested"], "hvac_kwh": df[f"floor{n}_hvac_kwh"],
            "in_comfort": df[f"floor{n}_in_comfort"], "mode": df[f"floor{n}_mode"],
            "preclamp_temp": df[f"floor{n}_preclamp_temp"], "cf_temp": df[f"floor{n}_cf_temp"],
            "controlled": df[f"floor{n}_controlled"],
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
