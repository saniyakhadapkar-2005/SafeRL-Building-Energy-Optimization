"""Safe-RL multi-floor building HVAC environment.

Changes versus the original project (see README "What changed"):

* Observation v3 appends the three per-floor target temperatures so the PPO
  agent can react to user-configured targets. ``observation_version="v2"``
  reproduces the original 11-feature layout so legacy v2 models can still be
  loaded and evaluated.
* Per-floor target temperatures (``set_targets``); ``target_temperature``
  still works as a scalar convenience property.
* Safety is now measured *before* the physical-bound clamp. The original code
  clamped temperatures to 18-30 °C and then checked them, so the temperature
  safety flag could never be False. The clamp (emergency HIGH) is retained as
  a last resort, but violations are recorded first.
* Counterfactual info: what each floor's temperature would have been if the
  raw (pre-Safety-Layer) actions had been applied for this one step.
* Per-floor HVAC energy and strict per-floor comfort/safety flags in ``info``.
* Synthetic (no-dataset) episodes advance the hour and use a diurnal outdoor
  temperature / occupancy / solar profile instead of values frozen for the
  whole episode.
* ``max_steps`` is configurable.

The action space (MultiDiscrete([4, 4, 4])), the physics, the energy model and
the reward formula are unchanged.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from safety.safety_layer import SafetyLayer, normalize_targets

NUM_FLOORS = 3
FLOOR_NAMES = ("Floor 1", "Floor 2", "Floor 3")
ACTION_NAMES = {0: "OFF", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}

SAFETY_MIN_C = 18.0
SAFETY_MAX_C = 30.0
COMFORT_MIN_C = 22.0
COMFORT_MAX_C = 25.0
DEFAULT_TARGET_C = 23.5
# Targets must keep a margin from the hard safety limits.
TARGET_MIN_C = SAFETY_MIN_C + 2.0
TARGET_MAX_C = SAFETY_MAX_C - 2.0

OBS_FEATURES_V2 = (
    "floor_1_temp", "floor_2_temp", "floor_3_temp", "outdoor_temp", "outdoor_humidity",
    "indoor_humidity", "occupancy", "electricity_price", "solar_generation",
    "building_area", "hour",
)
OBS_FEATURES_V3 = OBS_FEATURES_V2 + ("floor_1_target", "floor_2_target", "floor_3_target")

_OBS_LOW = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
                     SAFETY_MIN_C, SAFETY_MIN_C, SAFETY_MIN_C], dtype=np.float32)
_OBS_HIGH = np.array([45.0, 45.0, 45.0, 50.0, 100.0, 100.0, 100.0, 20.0, 200.0, 100000.0, 23.0,
                      SAFETY_MAX_C, SAFETY_MAX_C, SAFETY_MAX_C], dtype=np.float32)

_HVAC_POWER_KW = {0: 0.0, 1: 2.0, 2: 4.0, 3: 6.0}
_HVAC_TEMP_EFFECT = {0: 0.0, 1: 0.35, 2: 0.75, 3: 1.20}
_FAN_SPEED_PCT = {0: 0, 1: 35, 2: 65, 3: 100}


def validate_target(value: float) -> float:
    """Validate one target temperature and return it as ``float``."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Target temperature must be a finite number.")
    if not (TARGET_MIN_C <= value <= TARGET_MAX_C):
        raise ValueError(
            f"Target temperature {value:.1f}°C is outside the allowed range "
            f"{TARGET_MIN_C:.0f}–{TARGET_MAX_C:.0f}°C (hard safety limits are "
            f"{SAFETY_MIN_C:.0f}–{SAFETY_MAX_C:.0f}°C)."
        )
    return value


def occupancy_fraction(hour: float) -> float:
    """Simple weekday office occupancy profile (0-1). A simulation assumption."""
    h = int(hour) % 24
    if h < 7 or h >= 20:
        return 0.05
    if h < 9 or h >= 17:
        return 0.6
    return 1.0


class MultiFloorBuildingEnv(gym.Env):
    """Safe RL environment for a 3-floor building.

    Action: one HVAC level per floor, 0=OFF, 1=LOW, 2=MEDIUM, 3=HIGH.

    Observation (normalised to 0-1)::

        v2 (11): floor temps x3, outdoor temp, outdoor RH, indoor RH, occupancy,
                 electricity price, solar generation, building area, hour
        v3 (14): v2 + per-floor target temperatures x3
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data=None,
        initial_temp=None,
        max_steps: int = 168,
        observation_version: str = "v3",
        randomize_start: bool = False,
        target_temperatures: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if observation_version not in ("v2", "v3"):
            raise ValueError("observation_version must be 'v2' or 'v3'.")
        self.observation_version = observation_version
        self.obs_features = OBS_FEATURES_V3 if observation_version == "v3" else OBS_FEATURES_V2
        self.randomize_start = bool(randomize_start)

        self.data = data.reset_index(drop=True) if data is not None else None

        self.action_space = spaces.MultiDiscrete([4, 4, 4])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.obs_features),), dtype=np.float32
        )

        self.safety_min_temperature = SAFETY_MIN_C
        self.safety_max_temperature = SAFETY_MAX_C
        self.comfort_min_temperature = COMFORT_MIN_C
        self.comfort_max_temperature = COMFORT_MAX_C
        self.comfort_min_humidity = 40.0
        self.comfort_max_humidity = 60.0

        self.target_temperatures = np.full(NUM_FLOORS, DEFAULT_TARGET_C, dtype=np.float64)
        if target_temperatures is not None:
            self.set_targets(target_temperatures)

        self.initial_temps = self._as_floor_array(initial_temp if initial_temp is not None else 24.0)

        self.current_step = 0
        self._offset = 0  # first dataset row of the current episode (data mode)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        self.floor_temperatures = self.initial_temps.astype(np.float32).copy()
        self.humidity = 50.0
        self.outdoor_humidity = 50.0
        self.outdoor_temperature = 30.0
        self.occupancy = 50.0
        self.electricity_price = 8.0
        self.solar_generation = 10.0
        self.building_area = 2500.0
        self.hour = 12.0
        self.last_actions = np.zeros(NUM_FLOORS, dtype=np.int64)
        self.last_info: dict = {}

        # synthetic-episode parameters (only used when data is None)
        self._syn = {"temp": 30.0, "amp": 4.0, "hum": 60.0, "occ": 60.0, "solar": 20.0, "start_hour": 12}

    # ------------------------------------------------------------------
    # helpers / properties
    # ------------------------------------------------------------------
    @staticmethod
    def _as_floor_array(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(NUM_FLOORS, float(arr))
        arr = arr.reshape(-1)
        if arr.shape != (NUM_FLOORS,) or not np.all(np.isfinite(arr)):
            raise ValueError("Expected one finite value or three finite values (one per floor).")
        return arr

    @property
    def initial_temp(self) -> float:  # backwards compatible scalar view
        return float(np.mean(self.initial_temps))

    @property
    def target_temperature(self) -> float:
        """Mean target (scalar view kept for backwards compatibility)."""
        return float(np.mean(self.target_temperatures))

    @target_temperature.setter
    def target_temperature(self, value: float) -> None:
        self.set_targets(float(value))

    def set_targets(self, targets) -> None:
        """Set per-floor targets (scalar or 3 values). Raises ``ValueError`` if unsafe."""
        values = normalize_targets(targets)
        self.target_temperatures = np.array([validate_target(v) for v in values], dtype=np.float64)

    def set_initial_temperatures(self, temps) -> None:
        arr = self._as_floor_array(temps)
        if np.any(arr < self.safety_min_temperature) or np.any(arr > self.safety_max_temperature):
            raise ValueError("Initial temperatures must lie inside the safety limits.")
        self.initial_temps = arr

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "targets" in options:
            self.set_targets(options["targets"])
        if "initial_temps" in options:
            self.set_initial_temperatures(options["initial_temps"])

        self.current_step = 0
        self.humidity = 50.0
        self.last_actions = np.zeros(NUM_FLOORS, dtype=np.int64)

        start_temps = self.initial_temps.copy()
        if self.randomize_start and "initial_temps" not in options:
            start_temps = self.np_random.uniform(19.0, 29.0, size=NUM_FLOORS)
        self.floor_temperatures = start_temps.astype(np.float32)
        if self.randomize_start and "targets" not in options:
            tg = self.np_random.integers(40, 53, size=NUM_FLOORS) / 2.0  # 20.0 .. 26.0
            self.target_temperatures = tg.astype(np.float64)

        self._offset = 0
        if self.data is not None and self.randomize_start:
            # Training on a dataset: start each episode at a random row.
            latest_start = max(0, len(self.data) - self.max_steps)
            self._offset = int(self.np_random.integers(0, latest_start + 1))

        if self.data is None:
            self._syn = {
                "temp": float(self.np_random.uniform(18.0, 38.0)),
                "amp": float(self.np_random.uniform(2.0, 6.0)),
                "hum": float(self.np_random.uniform(30.0, 85.0)),
                "occ": float(self.np_random.uniform(20.0, 100.0)),
                "solar": float(self.np_random.uniform(0.0, 80.0)),
                "start_hour": int(self.np_random.integers(0, 24)),
            }
            self.electricity_price = float(self.np_random.uniform(4.0, 16.0))
            self.hour = float(self._syn["start_hour"])
            self._apply_synthetic_profile()

        self._load_environment_state()
        return self._get_observation(), {}

    def _apply_synthetic_profile(self) -> None:
        s = self._syn
        phase = math.sin(2.0 * math.pi * (self.hour - 9.0) / 24.0)  # peaks at 15:00
        self.outdoor_temperature = float(np.clip(s["temp"] + s["amp"] * phase, 0.0, 50.0))
        self.outdoor_humidity = float(np.clip(s["hum"] - 2.0 * s["amp"] * phase, 0.0, 100.0))
        self.occupancy = float(s["occ"] * occupancy_fraction(self.hour))
        self.solar_generation = float(s["solar"] * max(0.0, math.sin(math.pi * (self.hour - 6.0) / 12.0)))

    # ------------------------------------------------------------------
    # data row loading
    # ------------------------------------------------------------------
    def _load_environment_state(self) -> None:
        if self.data is None or len(self.data) == 0:
            return
        index = min(self._offset + self.current_step, len(self.data) - 1)
        row = self.data.iloc[index]
        self.outdoor_temperature = self._get_value(row, ["air_temperature", "outdoor_temperature"], 30.0)
        self.outdoor_humidity = self._get_value(
            row, ["relative_humidity_2m", "outdoor_humidity", "humidity"], self.outdoor_humidity)
        self.occupancy = self._get_value(row, ["occupancy"], 50.0)
        self.electricity_price = self._get_value(row, ["electricity_price"], 8.0)
        self.solar_generation = self._get_value(row, ["solar_generation"], 10.0)
        self.building_area = self._get_value(row, ["building_area_m2", "building_area"], 2500.0)
        if "hour" in row.index:
            self.hour = self._get_value(row, ["hour"], self.hour)

    @staticmethod
    def _get_value(row, possible_names, default):
        for name in possible_names:
            if name in row.index:
                value = row[name]
                if value is not None:
                    try:
                        value = float(value)
                        if np.isfinite(value):
                            return value
                    except (ValueError, TypeError):
                        pass
        return float(default)

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        raw = np.array(
            [
                *self.floor_temperatures,
                self.outdoor_temperature, self.outdoor_humidity, self.humidity,
                self.occupancy, self.electricity_price, self.solar_generation,
                self.building_area, self.hour,
                *self.target_temperatures,
            ],
            dtype=np.float32,
        )
        n = len(self.obs_features)
        raw = raw[:n]
        normalized = (raw - _OBS_LOW[:n]) / (_OBS_HIGH[:n] - _OBS_LOW[:n])
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # physics (unchanged from the original project, per-floor target added)
    # ------------------------------------------------------------------
    def _calculate_temperature_change(self, floor_index, action, current_temp=None):
        if current_temp is None:
            current_temp = float(self.floor_temperatures[floor_index])
        thermal_drift = (self.outdoor_temperature - current_temp) * 0.04
        occupancy_heat = (self.occupancy / 100.0) * 0.12
        floor_heat = floor_index * 0.04
        hvac_effect = _HVAC_TEMP_EFFECT[int(action)]
        if current_temp > self.target_temperatures[floor_index]:
            hvac_change = -hvac_effect
        else:
            hvac_change = hvac_effect
        new_temp = current_temp + thermal_drift + occupancy_heat + floor_heat + hvac_change
        return float(np.clip(new_temp, 15.0, 35.0))

    def _calculate_humidity_change(self, actions):
        average_hvac = float(np.mean(actions))
        occupancy_effect = (self.occupancy / 100.0) * 0.20
        outdoor_exchange_effect = (self.outdoor_humidity - self.humidity) * 0.03
        hvac_effect = average_hvac * 0.10
        natural_effect = (50.0 - self.humidity) * 0.02
        new_humidity = (self.humidity + natural_effect + occupancy_effect
                        + outdoor_exchange_effect - hvac_effect)
        return float(np.clip(new_humidity, 20.0, 80.0))

    def _area_factor(self) -> float:
        return float(np.sqrt(max(self.building_area, 100.0) / 2500.0))

    def _calculate_energy(self, actions):
        area_factor = self._area_factor()
        floor_hvac = [_HVAC_POWER_KW[int(a)] * area_factor for a in actions]
        hvac_energy = float(sum(floor_hvac))
        base_energy = 2.0 + (self.occupancy / 100.0) * 3.0 + (self.building_area / 2500.0) * 1.5
        total_energy = base_energy + hvac_energy
        solar_used = min(self.solar_generation, total_energy)
        grid_energy = max(total_energy - solar_used, 0.0)
        electricity_cost = grid_energy * self.electricity_price
        return {
            "hvac_energy_kwh": hvac_energy,
            "base_energy_kwh": float(base_energy),
            "total_energy_kwh": float(total_energy),
            "solar_used_kwh": float(solar_used),
            "grid_energy_kwh": float(grid_energy),
            "electricity_cost_inr": float(electricity_cost),
            "floor_hvac_energy_kwh": [float(x) for x in floor_hvac],
        }

    def _calculate_comfort(self) -> float:
        temperature_scores = []
        for temperature in self.floor_temperatures:
            if self.comfort_min_temperature <= temperature <= self.comfort_max_temperature:
                score = 100.0
            else:
                distance = min(abs(temperature - self.comfort_min_temperature),
                               abs(temperature - self.comfort_max_temperature))
                score = max(0.0, 100.0 - distance * 25.0)
            temperature_scores.append(score)
        temperature_score = float(np.mean(temperature_scores))
        if self.comfort_min_humidity <= self.humidity <= self.comfort_max_humidity:
            humidity_score = 100.0
        else:
            humidity_score = max(0.0, 100.0 - min(abs(self.humidity - self.comfort_min_humidity),
                                                  abs(self.humidity - self.comfort_max_humidity)) * 5.0)
        return float(0.7 * temperature_score + 0.3 * humidity_score)

    def _temps_safe(self, temps) -> list[bool]:
        return [bool(self.safety_min_temperature <= float(t) <= self.safety_max_temperature) for t in temps]

    def _calculate_reward(self, energy_data, comfort_score, safety_status, actions, override):
        """Unchanged formula. ``safety_status`` is now measured before the clamp."""
        comfort_reward = float(comfort_score) / 5.0
        energy_penalty = float(energy_data["total_energy_kwh"]) * 0.05
        cost_penalty = float(energy_data["electricity_cost_inr"]) * 0.005
        safety_penalty = 0.0 if safety_status else 20.0
        switching_penalty = float(np.sum(np.abs(np.asarray(actions) - self.last_actions))) * 0.1
        override_penalty = 0.1 if override else 0.0
        return float(comfort_reward - energy_penalty - cost_penalty - safety_penalty
                     - switching_penalty - override_penalty)

    def _get_timestamp(self) -> str:
        if self.data is not None and len(self.data) > 0:
            index = min(self._offset + self.current_step, len(self.data) - 1)
            row = self.data.iloc[index]
            if "timestamp" in row.index:
                return str(row["timestamp"])
        return str(self.current_step)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape != (NUM_FLOORS,):
            raise ValueError(f"Expected {NUM_FLOORS} HVAC actions, got shape {action.shape}.")
        action = np.clip(action, 0, 3)

        safety_layer = SafetyLayer(self.safety_min_temperature, self.safety_max_temperature)
        raw_actions = action.copy()

        # Safety Layer: always applied BEFORE anything reaches the physics.
        safe_actions, override, reasons = safety_layer.enforce_safety(
            state=self.floor_temperatures,
            raw_actions=raw_actions,
            occupancy=self.occupancy,
            target_temperature=self.target_temperatures,
        )
        safe_actions = np.asarray(safe_actions, dtype=np.int64)

        # Counterfactual: same state, raw agent actions, no Safety Layer (one step only).
        counterfactual_temps = [
            self._calculate_temperature_change(i, raw_actions[i]) for i in range(NUM_FLOORS)
        ]

        pre_clamp_temps = [
            self._calculate_temperature_change(i, safe_actions[i]) for i in range(NUM_FLOORS)
        ]
        floor_safety_ok = self._temps_safe(pre_clamp_temps)
        counterfactual_ok = self._temps_safe(counterfactual_temps)

        self.floor_temperatures = np.array(pre_clamp_temps, dtype=np.float32)
        self.humidity = self._calculate_humidity_change(safe_actions)

        # Post-action emergency clamp (kept from the original project).
        post_action_override = False
        for i in range(NUM_FLOORS):
            temperature = float(self.floor_temperatures[i])
            if temperature < self.safety_min_temperature or temperature > self.safety_max_temperature:
                side = "below" if temperature < self.safety_min_temperature else "above"
                bound = self.safety_min_temperature if side == "below" else self.safety_max_temperature
                self.floor_temperatures[i] = bound
                safe_actions[i] = 3
                reasons[i] = (f"Floor {i+1}: post-action temperature {temperature:.1f}°C {side} "
                              f"safety limit. Emergency HVAC set to HIGH.")
                post_action_override = True
        if post_action_override:
            override = True
        self.floor_temperatures = np.clip(
            self.floor_temperatures, self.safety_min_temperature, self.safety_max_temperature
        ).astype(np.float32)

        energy_data = self._calculate_energy(safe_actions)
        comfort_score = self._calculate_comfort()
        humidity_safe = 20.0 <= self.humidity <= 80.0
        safety_status = bool(all(floor_safety_ok) and humidity_safe)
        reward = self._calculate_reward(energy_data, comfort_score, safety_status, safe_actions, override)
        current_timestamp = self._get_timestamp()
        step_hour = float(self.hour)
        floor_in_comfort = [
            bool(self.comfort_min_temperature <= float(t) <= self.comfort_max_temperature)
            for t in self.floor_temperatures
        ]

        info_outdoor_temp = float(self.outdoor_temperature)
        info_outdoor_hum = float(self.outdoor_humidity)
        info_occupancy = float(self.occupancy)
        info_price = float(self.electricity_price)
        info_solar = float(self.solar_generation)
        info_area = float(self.building_area)

        self.current_step += 1
        dataset_finished = self.data is not None and (self._offset + self.current_step) >= len(self.data)
        terminated = bool(dataset_finished)
        truncated = bool(self.current_step >= self.max_steps)

        hvac_modes = []
        for floor_temp, floor_action, target in zip(self.floor_temperatures, safe_actions,
                                                    self.target_temperatures):
            if floor_action == 0:
                hvac_modes.append("OFF")
            elif floor_temp > target:
                hvac_modes.append("COOLING")
            else:
                hvac_modes.append("HEATING")

        info = {
            "ppo_action": raw_actions.tolist(),
            "safe_action": safe_actions.tolist(),
            "safety_override": bool(override),
            "post_action_override": bool(post_action_override),
            "safety_reason": reasons,
            "floor_temperatures": self.floor_temperatures.tolist(),
            "floor_pre_clamp_temperatures": [float(t) for t in pre_clamp_temps],
            "floor_counterfactual_temperatures": [float(t) for t in counterfactual_temps],
            "floor_safety_ok": floor_safety_ok,
            "floor_safety_ok_without_layer": counterfactual_ok,
            "floor_in_comfort": floor_in_comfort,
            "floor_targets": self.target_temperatures.tolist(),
            "indoor_temperature": float(np.mean(self.floor_temperatures)),
            "floor_1_temperature": float(self.floor_temperatures[0]),
            "floor_2_temperature": float(self.floor_temperatures[1]),
            "floor_3_temperature": float(self.floor_temperatures[2]),
            "outdoor_temperature": info_outdoor_temp,
            "outdoor_humidity": info_outdoor_hum,
            "humidity": float(self.humidity),
            "occupancy": info_occupancy,
            "electricity_price": info_price,
            "solar_generation": info_solar,
            "building_area": info_area,
            "hour": step_hour,
            "hvac_levels": safe_actions.tolist(),
            "hvac_modes": hvac_modes,
            "fan_speed_pct": [_FAN_SPEED_PCT[int(a)] for a in safe_actions],
            "comfort_score": float(comfort_score),
            "comfort_status": bool(comfort_score >= 80.0),
            "safety_status": safety_status,
            "target_temperature": self.target_temperature,
            "reward": float(reward),
            "timestamp": current_timestamp,
            **energy_data,
        }

        self.last_actions = safe_actions.copy()
        self.last_info = info

        if not terminated and not truncated:
            if self.data is None:
                self.hour = float((self.hour + 1) % 24)
                self._apply_synthetic_profile()
            else:
                self._load_environment_state()

        return self._get_observation(), reward, terminated, truncated, info
