"""Streamlit UI helpers: styling, version-tolerant widgets, small components."""
from __future__ import annotations

import inspect
from typing import Optional

import pandas as pd
import streamlit as st

from environment.building_env import ACTION_NAMES, COMFORT_MAX_C, COMFORT_MIN_C, SAFETY_MAX_C, SAFETY_MIN_C

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.app-hero {padding: 1rem 1.4rem; border-radius: 16px; margin-bottom: .8rem;
  background: linear-gradient(115deg,rgba(18,45,77,.96),rgba(53,37,91,.94)); border: 1px solid #394d7c;}
.app-hero h1 {margin: 0; font-size: 1.7rem; color: #f5f8ff;}
.app-hero p {margin: .35rem 0 0 0; color: #c5d5f4; font-size: .92rem;}
.badge {display:inline-block; border-radius:999px; padding:.18rem .7rem; font-size:.78rem; font-weight:700; margin-right:.35rem;}
.badge-ok {background:#14532d; color:#bbf7d0;} .badge-warn {background:#713f12; color:#fde68a;}
.badge-err {background:#7f1d1d; color:#fecaca;} .badge-info {background:#1e3a5f; color:#bfdbfe;}
.floor-card {padding: .9rem 1rem; border-radius: 14px; background: linear-gradient(150deg,#19263d,#202b46);
  border: 1px solid #344867; height: 100%;}
.floor-card.selected {border: 2px solid #4cc9f0;}
.floor-title {font-size: 1.02rem; font-weight: 750; color: #e8f2ff;}
.floor-temp {font-size: 2rem; font-weight: 800; color: #62e6ce; margin: .15rem 0;}
.muted {color: #aab8d2; font-size: .8rem;}
.kv {display:flex; justify-content:space-between; font-size:.86rem; padding:.12rem 0; color:#dbe6fb;}
.kv b {color:#fff;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Version-tolerant keyword arguments (Streamlit renamed use_container_width -> width="stretch")
# ----------------------------------------------------------------------
def _stretch(fn) -> dict:
    params = inspect.signature(fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def button(label: str, **kwargs):
    return st.button(label, **{**_stretch(st.button), **kwargs})


def show_chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, **_stretch(st.plotly_chart))


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **{**_stretch(st.dataframe), **kwargs})


def download_button(label: str, data, file_name: str, mime: str, **kwargs):
    return st.download_button(label, data, file_name, mime, **{**_stretch(st.download_button), **kwargs})


# ----------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------
def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def flash_messages() -> None:
    for kind, text in st.session_state.pop("flash", []):
        {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(text)


def push_flash(kind: str, text: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, text))


def fmt(value: Optional[float], spec: str = ".1f", unit: str = "", missing: str = "—") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return missing
    return f"{value:{spec}}{unit}"


def floor_card_html(n: int, row: Optional[dict], selected: bool, cumulative_kwh: Optional[float]) -> str:
    cls = "floor-card selected" if selected else "floor-card"
    if row is None:
        return (f'<div class="{cls}"><div class="floor-title">🏢 Floor {n}</div>'
                '<div class="floor-temp">—</div><div class="muted">No simulation state yet. '
                'Start a simulation to see this floor.</div></div>')
    t, target = float(row[f"floor{n}_temp"]), float(row[f"floor{n}_target"])
    diff = t - target
    action, requested = row[f"floor{n}_action_name"], ACTION_NAMES[int(row[f"floor{n}_requested"])]
    controlled = bool(row[f"floor{n}_controlled"])
    in_comfort = bool(row[f"floor{n}_in_comfort"])
    pre = float(row[f"floor{n}_preclamp_temp"])
    safe_ok = SAFETY_MIN_C <= pre <= SAFETY_MAX_C
    comfort_badge = badge("Comfort range", "ok") if in_comfort else badge(
        "Below comfort" if t < COMFORT_MIN_C else "Above comfort", "warn")
    safety_badge = badge("Within safety limits", "ok") if safe_ok else badge("SAFETY VIOLATION", "err")
    req_line = (f"{requested}" if controlled else f"{requested} <span class='muted'>(not agent-controlled)</span>")
    return (
        f'<div class="{cls}"><div class="floor-title">🏢 Floor {n}</div>'
        f'<div class="floor-temp">{t:.1f}°C</div><div class="muted">Indoor temperature · simulated</div><hr style="border-color:#344867;margin:.5rem 0">'
        f'<div class="kv"><span>Target</span><b>{target:.1f}°C ({diff:+.1f}°C)</b></div>'
        f'<div class="kv"><span>Requested action</span><b>{req_line}</b></div>'
        f'<div class="kv"><span>Applied HVAC action</span><b>{action}</b></div>'
        f'<div class="kv"><span>HVAC mode</span><b>{row[f"floor{n}_mode"]}</b></div>'
        f'<div class="kv"><span>HVAC energy (this step)</span><b>{float(row[f"floor{n}_hvac_kwh"]):.2f} kWh</b></div>'
        f'<div class="kv"><span>HVAC energy (so far)</span><b>{fmt(cumulative_kwh, ".1f", " kWh")}</b></div>'
        f'<div style="margin-top:.5rem">{comfort_badge}{safety_badge}</div></div>')
