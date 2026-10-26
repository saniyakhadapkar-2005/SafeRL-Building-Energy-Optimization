"""Reusable dashboard sections shared by several pages."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from dashboard import state, ui
from dashboard.charts import comfort_bar_chart, empty_figure
from simulation import metrics
from simulation.runner import (CONTROLLER_LABELS, STATE_FINISHED, STATE_PAUSED, STATE_READY, STATE_RUNNING)
from utils.weather import STATUS_LIVE, STATUS_SIMULATED, STATUS_STALE

STATE_LABEL = {STATE_READY: ("Ready", "info"), STATE_RUNNING: ("Running", "ok"),
               STATE_PAUSED: ("Paused", "warn"), STATE_FINISHED: ("Finished", "info")}


def weather_badge(snap) -> str:
    if snap is None:
        return ui.badge("Weather: not loaded", "warn")
    return {STATUS_LIVE: ui.badge("Weather: LIVE (Open-Meteo)", "ok"),
            STATUS_STALE: ui.badge("Weather: STALE live reading", "warn"),
            STATUS_SIMULATED: ui.badge("Weather: SIMULATED fallback (not real)", "err")}[snap.status]


def model_badge(status) -> str:
    if status.ready:
        return ui.badge("PPO model: ready" + (" (legacy v2)" if status.is_legacy else ""), "ok" if not status.is_legacy else "warn")
    return ui.badge(f"PPO model: {status.state}", "err")


def header() -> None:
    s = state.settings()
    _, mstatus = state.get_model()
    snap = st.session_state.get("weather")
    src = s["weather_source"]
    wb = weather_badge(snap) if src == "live" else ui.badge(f"Weather source: {state.WEATHER_SOURCES[src]}", "info")
    controller = "PPO agent" if s["controller"] == "ppo" else "Baseline thermostat (not RL)"
    st.markdown(
        f'<div class="app-hero"><h1>🏢 Safe RL Building Energy Optimization</h1>'
        f'<p>📍 <b>{state.location().label}</b> '
        f'({state.location().latitude:.3f}, {state.location().longitude:.3f}) · Controller selected: <b>{controller}</b> · '
        f'Floor view: <b>{state.floor_view()}</b></p>'
        f'<div style="margin-top:.5rem">{wb}{model_badge(mstatus)}'
        f'{ui.badge("Indoor values: SIMULATED", "info")}</div></div>', unsafe_allow_html=True)
    if src == "live" and snap is not None and snap.warning:
        st.warning(snap.warning)


def sim_controls(prefix: str) -> None:
    """Start / Pause / Resume / Step / Reset / Refresh weather. ``prefix`` keeps widget keys unique per page."""
    sim = state.get_sim()
    stt = sim.state if sim else None
    c = st.columns([1.3, 1, 1, 1, 1, 1.4])
    can_start = sim is None or stt == STATE_READY
    scope = state.floor_view()
    label = "▶ Start — entire building" if scope == "All Floors" else f"▶ Start — {scope}"
    c[0].button(label, key=f"{prefix}_start", type="primary", disabled=not can_start, on_click=state.cb_start,
                **ui._stretch(st.button))
    c[1].button("⏸ Pause", key=f"{prefix}_pause", disabled=stt != STATE_RUNNING, on_click=state.cb_pause,
                **ui._stretch(st.button))
    c[2].button("⏯ Resume", key=f"{prefix}_resume", disabled=stt != STATE_PAUSED, on_click=state.cb_resume,
                **ui._stretch(st.button))
    c[3].button("⏭ One step", key=f"{prefix}_step", disabled=stt in (STATE_RUNNING, STATE_FINISHED),
                on_click=state.cb_step_once, **ui._stretch(st.button))
    c[4].button("↺ Reset", key=f"{prefix}_reset", disabled=sim is None, on_click=state.cb_reset,
                **ui._stretch(st.button))
    c[5].button("🌦 Refresh weather", key=f"{prefix}_wx", on_click=state.cb_refresh_weather,
                **ui._stretch(st.button))
    if state.settings_drifted():
        st.info("Settings changed after this run was created. Press **Reset** and **Start** to apply them "
                "(target temperatures can be applied live from the Floor Controls page).")
    if sim is not None:
        meta = st.session_state.sim_meta or {}
        badge_text, kind = STATE_LABEL[sim.state]
        st.markdown(
            f'{ui.badge(badge_text, kind)} Step <b>{sim.current_step}</b> / {sim.total_steps} · '
            f'{meta.get("controller", "")} · {meta.get("scope", "")} · weather: {meta.get("weather_source", "")}',
            unsafe_allow_html=True)
        st.progress(sim.current_step / sim.total_steps)
    else:
        st.caption("No simulation created yet. Press Start to run the real environment step by step "
                   "(1 step = 1 simulated hour).")


def kpi_row(df: Optional[pd.DataFrame], sim) -> None:
    meta = st.session_state.get("sim_meta") or {}
    snap = st.session_state.get("weather")
    has = df is not None and len(df) > 0
    s = metrics.summarize(df) if has else None
    cols = st.columns(7)
    if has:
        outdoor, label = float(df["outdoor_temperature"].iloc[-1]), "Outdoor (sim step input)"
    elif snap is not None:
        outdoor, label = snap.current.get("temperature_c"), f"Outdoor now ({snap.status})"
    else:
        outdoor, label = None, "Outdoor temperature"
    cols[0].metric(label, ui.fmt(outdoor, ".1f", " °C"))
    cols[1].metric("Avg indoor (simulated)", ui.fmt(float(df["indoor_temperature"].iloc[-1]) if has else None, ".1f", " °C"))
    cols[2].metric("Energy used", ui.fmt(s["total_energy_kwh"] if has else None, ".1f", " kWh"))
    tariff = meta.get("tariff", state.settings()["tariff"])
    cols[3].metric("Est. cost", ui.fmt(s["total_cost_inr"] if has else None, ".0f", " ₹"),
                   help=f"Estimated at the configured flat tariff of ₹{tariff:.2f}/kWh (not live pricing, not a bill).")
    cols[4].metric("Comfort", ui.fmt(s["comfort_pct"] if has else None, ".1f", " %"),
                   help=("Floor-steps with temperature in 22–25 °C ÷ all floor-steps evaluated: "
                         f"{s['comfort_num']}/{s['comfort_den']}") if has else "No steps evaluated yet.")
    cols[5].metric("Safety", ui.fmt(s["safety_pct"] if has else None, ".1f", " %"),
                   help=("Floor-steps within 18–30 °C after the Safety Layer (before emergency clamp) ÷ all "
                         f"floor-steps: {s['safety_num']}/{s['safety_den']}") if has else "No steps evaluated yet.")
    cols[6].metric("Step", f"{sim.current_step} / {sim.total_steps}" if sim else "—")


def floor_cards(df: Optional[pd.DataFrame]) -> None:
    chosen = state.selected_floors()
    st.markdown("#### Floor state" + ("" if len(chosen) == 3 else f" — {state.floor_view()}"))
    row = df.iloc[-1].to_dict() if df is not None and len(df) else None
    cols = st.columns(len(chosen))
    for col, n in zip(cols, chosen):
        cum = float(df[f"floor{n}_hvac_kwh"].sum()) if row is not None else None
        col.markdown(ui.floor_card_html(n, row, selected=len(chosen) == 1, cumulative_kwh=cum), unsafe_allow_html=True)
