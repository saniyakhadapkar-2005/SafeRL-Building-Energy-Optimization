"""Rule-based comfort and hard-safety filter for three-floor HVAC actions.

Every HVAC action requested by the PPO agent (or any other controller) is
passed through :meth:`SafetyLayer.enforce_safety` *before* it is applied to the
environment.

Backward compatible with the original API: ``target_temperature`` may be a
single number (same target for every floor) or a sequence with one target per
floor.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

NUM_FLOORS = 3
TargetLike = Union[float, int, Sequence[float]]


def normalize_targets(target_temperature: TargetLike, n_floors: int = NUM_FLOORS) -> list[float]:
    """Return one finite target per floor from a scalar or a sequence."""
    try:
        targets = [float(target_temperature)] * n_floors
    except TypeError:
        targets = [float(t) for t in target_temperature]
    if len(targets) != n_floors:
        raise ValueError(f"Expected 1 or {n_floors} target temperatures, got {len(targets)}.")
    if not all(math.isfinite(t) for t in targets):
        raise ValueError("Target temperatures must be finite numbers.")
    return targets


class SafetyLayer:
    def __init__(self, temp_min: float = 18.0, temp_max: float = 30.0):
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        if self.temp_min >= self.temp_max:
            raise ValueError("temp_min must be smaller than temp_max.")

    def enforce_safety(self, state, raw_actions, occupancy=50.0, target_temperature: TargetLike = 23.5):
        """Return ``(safe_actions, overridden, reasons)``.

        ``state`` is the current temperature of each floor, ``raw_actions`` the
        requested HVAC level per floor (0=OFF .. 3=HIGH).
        """
        actions = [max(0, min(3, int(a))) for a in raw_actions]
        if len(actions) != NUM_FLOORS or len(state) != NUM_FLOORS:
            raise ValueError("Expected exactly 3 temperatures and 3 HVAC actions.")
        targets = normalize_targets(target_temperature)
        reasons = ["No override"] * NUM_FLOORS
        overridden = False
        for i, (temperature, action) in enumerate(zip(state, actions)):
            temperature = float(temperature)
            if not math.isfinite(temperature):
                # A non-finite reading can never be trusted: emergency HIGH.
                actions[i] = 3
                reasons[i] = f"Floor {i+1}: non-finite temperature reading; emergency HIGH."
                overridden = True
                continue
            target = targets[i]
            # Hard safety boundary always wins.
            if temperature < self.temp_min or temperature > self.temp_max:
                actions[i] = 3
                direction = "below" if temperature < self.temp_min else "above"
                reasons[i] = f"Floor {i+1}: {temperature:.1f}°C {direction} hard safety bound; emergency HIGH."
                overridden = True
            # Target-aware comfort guard (unchanged rules from the original project).
            elif temperature > target + 1.0 and action < 2:
                actions[i] = 2
                reasons[i] = (f"Floor {i+1}: {temperature:.1f}°C is >1°C above target "
                              f"{target:.1f}°C; raised to MEDIUM cooling.")
                overridden = True
            elif temperature < target - 1.0 and action < 2:
                actions[i] = 2
                reasons[i] = (f"Floor {i+1}: {temperature:.1f}°C is >1°C below target "
                              f"{target:.1f}°C; raised to MEDIUM heating.")
                overridden = True
            elif temperature >= self.temp_max - 1.5 and action < 3:
                actions[i] = 3
                reasons[i] = f"Floor {i+1}: near upper hard safety bound; raised to HIGH."
                overridden = True
            elif temperature <= self.temp_min + 1.5 and action < 3:
                actions[i] = 3
                reasons[i] = f"Floor {i+1}: near lower hard safety bound; raised to HIGH."
                overridden = True
            elif 22.0 <= temperature <= 25.0 and float(occupancy) < 15.0 and action > 1:
                actions[i] = 1
                reasons[i] = f"Floor {i+1}: comfortable temperature and low occupancy; capped at LOW."
                overridden = True
        return actions, overridden, reasons
