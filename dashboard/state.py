"""Session state, service wiring and simulation lifecycle for the dashboard.

No chart or layout code lives here. Simulation logic itself is in ``simulation/``.
"""
from __future__ import annotations

import copy
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from environment.building_env import NUM_FLOORS, validate_target
from rl.model_io import (LEGACY_MODEL_PATH, MODEL_DIR, MODEL_PATH, TRAIN_STATUS_PATH, ModelStatus, load_model,
                         read_training_status)
from simulation.runner import (CONTROLLER_BASELINE, CONTROLLER_LABELS, CONTROLLER_PPO, STATE_FINISHED,
                               STATE_PAUSED, STATE_READY, STATE_RUNNING, SimConfig, SimulationError,
                               SimulationSession)
from simulation.scenario import (ScenarioConfig, ScenarioError, from_historical_frame, from_weather_snapshot)
from utils.preprocessing import data_files_present, load_prepared_data
from utils.weather import (CITY_PRESETS, Location, WeatherError, WeatherService, WeatherSnapshot,
                           preset_location, simulated_weather)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LIVE_INTERVAL_S = 0.8
START_TEMPS = {"Cool": 21.0, "Comfortable": 23.5, "Warm": 27.0, "Hot": 29.0}
OCCUPANCY = {"Quiet": 10.0, "Typical": 25.0, "Busy": 40.0, "Full": 60.0}
FLOOR_VIEWS = ["All Floors", "Floor 1", "Floor 2", "Floor 3"]
WEATHER_SOURCES = {
    "live": "Live weather (Open-Meteo)",
    "simulated": "Simulated sample weather (not real)",
    "historical": "Historical CSV dataset (data/ folder)",
}

DEFAULT_SETTINGS = {
    "location_choice": "Mumbai", "location": None, "weather_source": "live",
    "link_targets": True, "targets": [23.5, 23.5, 23.5], "start_condition": "Warm",
    "steps": 24, "occupancy": "Typical", "tariff": 8.0, "pv_kw": 0.0, "area": 2500.0,
    "controller": CONTROLLER_PPO, "speed": 1, "weather_ttl_min": 10,
}


# ----------------------------------------------------------------------
# state helpers
# ----------------------------------------------------------------------
def init_state() -> None:
    if "settings" not in st.session_state:
        s = copy.deepcopy(DEFAULT_SETTINGS)
        s["location"] = preset_location("Mumbai")
        st.session_state.settings = s
    st.session_state.setdefault("sim", None)
    st.session_state.setdefault("sim_meta", None)
    st.session_state.setdefault("weather", None)
    st.session_state.setdefault("weather_key", None)
    st.session_state.setdefault("train_proc", None)


def settings() -> dict:
    return st.session_state.settings


def reset_settings() -> None:
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s["location"] = preset_location("Mumbai")
    st.session_state.settings = s


def floor_view() -> str:
    return st.session_state.get("floor_view", "All Floors")


def selected_floors() -> list[int]:
    v = floor_view()
    return list(range(1, NUM_FLOORS + 1)) if v == "All Floors" else [int(v.split()[1])]


def view_scope() -> Optional[int]:
    v = floor_view()
    return None if v == "All Floors" else int(v.split()[1]) - 1


def get_sim() -> Optional[SimulationSession]:
    return st.session_state.get("sim")


def is_running() -> bool:
    sim = get_sim()
    return sim is not None and sim.state == STATE_RUNNING


def interval_if_running() -> Optional[float]:
    return LIVE_INTERVAL_S if is_running() else None


def run_live(render_fn) -> None:
    """Render ``render_fn`` as an auto-refreshing fragment only while the simulation runs."""
    if is_running():
        st.fragment(run_every=LIVE_INTERVAL_S)(render_fn)()
    else:
        render_fn()


# ----------------------------------------------------------------------
# services
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def weather_service() -> WeatherService:
    return WeatherService()


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def _cached_model(key: tuple):
    return load_model()


def get_model():
    """Returns ``(model_or_None, ModelStatus)``; reloads automatically when the model file changes."""
    return _cached_model((_mtime(MODEL_PATH), _mtime(LEGACY_MODEL_PATH)))


@st.cache_resource(show_spinner="Loading dataset…")
def _historical_frame(key: float) -> pd.DataFrame:
    return load_prepared_data(DATA_DIR, sample_size=100000)


def historical_available() -> bool:
    return data_files_present(DATA_DIR)


def location() -> Location:
    return settings()["location"]


def _location_key(loc: Location) -> tuple:
    return (round(loc.latitude, 3), round(loc.longitude, 3))


def refresh_weather(force: bool = False) -> WeatherSnapshot:
    """Fetch (or reuse cached) live weather for the selected location. Never raises."""
    svc = weather_service()
    svc.ttl_seconds = int(settings()["weather_ttl_min"]) * 60
    loc = location()
    snap = svc.get_weather(loc, force=force)
    st.session_state.weather = snap
    st.session_state.weather_key = _location_key(loc)
    return snap


def ensure_weather() -> WeatherSnapshot:
    snap = st.session_state.get("weather")
    if snap is None or st.session_state.get("weather_key") != _location_key(location()):
        snap = refresh_weather(force=False)
    return snap


# ----------------------------------------------------------------------
# simulation lifecycle
# ----------------------------------------------------------------------
def _config_signature() -> tuple:
    s = settings()
    return (s["steps"], s["occupancy"], s["tariff"], s["pv_kw"], s["area"], s["controller"],
            s["weather_source"], s["start_condition"], _location_key(location()), view_scope())


def scenario_for_settings() -> tuple[pd.DataFrame, WeatherSnapshot | None, str]:
    s = settings()
    cfg = ScenarioConfig(steps=int(s["steps"]), occupancy_peak=OCCUPANCY[s["occupancy"]],
                         tariff_inr_per_kwh=float(s["tariff"]), pv_capacity_kw=float(s["pv_kw"]),
                         building_area_m2=float(s["area"])).validate()
    src = s["weather_source"]
    if src == "historical":
        if not historical_available():
            raise ScenarioError("Historical dataset not found. Place train.csv, weather_train.csv and "
                                "building_metadata.csv in the data/ folder, or pick another weather source.")
        frame = _historical_frame(max(_mtime(DATA_DIR / "train.csv"), 1.0))
        return from_historical_frame(frame, cfg), None, "Historical CSV dataset (not live)"
    if src == "simulated":
        snap = simulated_weather(location())
    else:
        snap = ensure_weather()
    return from_weather_snapshot(snap, cfg), snap, snap.source


def create_session() -> SimulationSession:
    s = settings()
    model, status = get_model()
    controller = s["controller"]
    if controller == CONTROLLER_PPO and not status.ready:
        raise SimulationError(f"PPO model not ready: {status.message}")
    targets = tuple(validate_target(t) for t in s["targets"])
    start = START_TEMPS[s["start_condition"]]
    scenario, snap, source = scenario_for_settings()
    obs_version = status.observation_version if controller == CONTROLLER_PPO else "v3"
    cfg = SimConfig(targets=targets, start_temps=(start,) * NUM_FLOORS, controller=controller,
                    scope=view_scope(), seed=42, observation_version=obs_version)
    sim = SimulationSession(scenario, cfg, model=model if controller == CONTROLLER_PPO else None)
    st.session_state.sim_meta = {
        "signature": _config_signature(), "location": location().label, "weather_source": source,
        "weather_status": snap.status if snap else "historical", "controller": CONTROLLER_LABELS[controller],
        "scope": "All floors" if cfg.scope is None else f"Floor {cfg.scope + 1} only",
        "tariff": float(s["tariff"]), "pv_kw": float(s["pv_kw"]), "model_note": status.message if controller == CONTROLLER_PPO else "",
        "created": datetime.now().strftime("%H:%M:%S"),
    }
    return sim


def _safe(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (SimulationError, ScenarioError, WeatherError, ValueError) as exc:
            st.session_state.setdefault("flash", []).append(("error", str(exc)))
        except Exception as exc:  # never crash the whole app from a button callback
            st.session_state.setdefault("flash", []).append(("error", f"Unexpected error: {type(exc).__name__}: {exc}"))
    return wrapper


@_safe
def cb_start() -> None:
    sim = get_sim()
    if sim is not None and sim.finished:
        st.session_state.flash = [("warning", "The simulation has finished. Press Reset to run again.")]
        return
    if sim is not None and sim.state == STATE_RUNNING:
        return  # guard against duplicate starts on reruns
    if sim is None or (sim.current_step == 0 and st.session_state.sim_meta
                       and st.session_state.sim_meta["signature"] != _config_signature()):
        st.session_state.sim = create_session()
    st.session_state.sim.start()


def cb_pause() -> None:
    sim = get_sim()
    if sim:
        sim.pause()


def cb_resume() -> None:
    sim = get_sim()
    if sim and sim.state == STATE_PAUSED:
        sim.start()


@_safe
def cb_step_once() -> None:
    if get_sim() is None:
        st.session_state.sim = create_session()
    sim = get_sim()
    if sim.state in (STATE_READY, STATE_PAUSED):
        sim.state = STATE_PAUSED
        sim.advance(1)
        if sim.finished:
            return


def cb_reset() -> None:
    st.session_state.sim = None
    st.session_state.sim_meta = None


@_safe
def cb_refresh_weather() -> None:
    snap = refresh_weather(force=True)
    kind = "success" if snap.is_live else "warning"
    msg = (f"Live weather updated for {location().label}." if snap.is_live
           else snap.warning or "Live weather unavailable.")
    st.session_state.setdefault("flash", []).append((kind, msg))


@_safe
def cb_apply_targets_live() -> None:
    sim = get_sim()
    if sim is None:
        return
    targets = tuple(validate_target(t) for t in settings()["targets"])
    sim.env.set_targets(list(targets))
    sim.config = replace(sim.config, targets=targets)
    st.session_state.setdefault("flash", []).append(("success", "New target temperatures applied from the next step."))


def settings_drifted() -> bool:
    meta, sim = st.session_state.get("sim_meta"), get_sim()
    return bool(sim is not None and meta and meta["signature"] != _config_signature())


def run_clock() -> None:
    """Advance the real environment while the simulation is running (called from a fragment)."""
    sim = get_sim()
    if sim is None or sim.state != STATE_RUNNING:
        return
    try:
        sim.advance(int(settings()["speed"]))
    except Exception as exc:
        st.session_state.setdefault("flash", []).append(("error", f"Simulation stopped: {type(exc).__name__}: {exc}"))
        st.rerun()
    if sim.finished:
        st.rerun()


# ----------------------------------------------------------------------
# PPO training in a separate process
# ----------------------------------------------------------------------
STALE_AFTER_S = 120


def training_state() -> dict:
    """Training status from the status file, with stale-run detection."""
    st_ = read_training_status()
    if st_.get("state") == "running":
        beat = float(st_.get("heartbeat", 0))
        proc = st.session_state.get("train_proc")
        alive = proc is not None and proc.poll() is None
        if not alive and time.time() - beat > STALE_AFTER_S:
            st_["state"] = "stale"
    return st_


def start_training(timesteps: int) -> None:
    cur = training_state()
    proc = st.session_state.get("train_proc")
    if cur.get("state") == "running" and proc is not None and proc.poll() is None:
        raise RuntimeError("Training is already running.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Mark as running first so a rerun can never launch a second process.
    TRAIN_STATUS_PATH.write_text(
        '{"state": "running", "timesteps": 0, "total": %d, "heartbeat": %f}' % (int(timesteps), time.time()),
        encoding="utf-8")
    log = open(MODEL_DIR / "training.log", "w", encoding="utf-8")
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    st.session_state.train_proc = subprocess.Popen(
        [sys.executable, "-m", "rl.train", "--timesteps", str(int(timesteps))],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, **kwargs)


def clear_model_cache() -> None:
    _cached_model.clear()
