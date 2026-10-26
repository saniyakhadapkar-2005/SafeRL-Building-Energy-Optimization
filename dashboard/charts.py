"""Plotly figure builders. Pure functions: DataFrame in, Figure out (no Streamlit).

All charts use the real simulation / evaluation tables. Empty or missing data
returns a figure with an explanatory message instead of raising.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from environment.building_env import (ACTION_NAMES, COMFORT_MAX_C, COMFORT_MIN_C, NUM_FLOORS,
                                      SAFETY_MAX_C, SAFETY_MIN_C)
from simulation.metrics import comfort_ratio
from simulation.runner import to_long_floor_frame

FLOOR_COLORS = {1: "#4cc9f0", 2: "#f9c74f", 3: "#90be6d"}
OUTDOOR_COLOR = "#f8961e"
TEMPLATE = "plotly_dark"
ACTION_COLORS = ["#3a4256", "#4cc9f0", "#f9c74f", "#f94144"]  # OFF, LOW, MEDIUM, HIGH


def _layout(fig: go.Figure, title: str, height: int = 380, ytitle: Optional[str] = None,
            xtitle: Optional[str] = None) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.01), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    return fig


def empty_figure(message: str, title: str = "", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#aab8d2"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, height)


def _floors(floors: Optional[Iterable[int]]) -> list[int]:
    chosen = sorted(set(floors)) if floors is not None else list(range(1, NUM_FLOORS + 1))
    return [f for f in chosen if 1 <= f <= NUM_FLOORS]


def _xcol(df: pd.DataFrame) -> tuple[str, str]:
    if "timestamp" in df.columns and pd.to_datetime(df["timestamp"], errors="coerce").notna().all():
        return "timestamp", "Time"
    return "step", "Simulation step (1 step = 1 hour)"


def _has_data(df) -> bool:
    return df is not None and len(df) > 0


def align_outdoor(sim_df: pd.DataFrame, outdoor_df: pd.DataFrame, value_col: str,
                  tolerance: str = "90min") -> pd.DataFrame:
    """Attach an external outdoor series (any sampling interval) to simulation timestamps.

    Uses nearest-time matching within ``tolerance``; unmatched rows stay NaN so the
    chart shows a gap instead of an invented value.
    """
    left = sim_df[["timestamp"]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"])
    right = outdoor_df[["timestamp", value_col]].copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"])
    right = right.dropna().sort_values("timestamp")
    merged = pd.merge_asof(left.sort_values("timestamp"), right, on="timestamp", direction="nearest",
                           tolerance=pd.Timedelta(tolerance))
    return merged


# ----------------------------------------------------------------------
# A. floor-wise indoor temperature
# ----------------------------------------------------------------------
def temperature_chart(df: pd.DataFrame, floors=None, show_bands: bool = True) -> go.Figure:
    title = "Indoor temperature by floor (simulated)"
    if not _has_data(df):
        return empty_figure("No simulation data yet. Start a simulation to see temperatures.", title)
    long = to_long_floor_frame(df)
    xcol, xtitle = _xcol(df)
    long = long.rename(columns={"timestamp": "timestamp"})
    xs = "timestamp" if xcol == "timestamp" else "step"
    fig = go.Figure()
    chosen = _floors(floors)
    values = []
    for n in chosen:
        f = long[long["floor_number"] == n]
        values.extend(f["temperature"].tolist())
        values.extend(f["target"].tolist())
        fig.add_trace(go.Scatter(
            x=f[xs], y=f["temperature"], mode="lines+markers", name=f"Floor {n} temperature",
            line=dict(color=FLOOR_COLORS[n], width=2.5), marker=dict(size=5),
            customdata=np.stack([f["action_name"], f["step"]], axis=-1),
            hovertemplate=(f"<b>Floor {n}</b><br>Time: %{{x}}<br>Step: %{{customdata[1]}}"
                           "<br>Temperature: %{y:.2f} °C<br>HVAC action: %{customdata[0]}<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=f[xs], y=f["target"], mode="lines", name=f"Floor {n} target",
            line=dict(color=FLOOR_COLORS[n], width=1.5, dash="dash"), opacity=0.8,
            hovertemplate=f"Floor {n} target: %{{y:.1f}} °C<extra></extra>"))
    lo = min(SAFETY_MIN_C - 1.5, min(values) - 0.5) if values else SAFETY_MIN_C - 1.5
    hi = max(SAFETY_MAX_C + 1.5, max(values) + 0.5) if values else SAFETY_MAX_C + 1.5
    if show_bands:
        fig.add_hrect(y0=COMFORT_MIN_C, y1=COMFORT_MAX_C, fillcolor="#2ec4b6", opacity=0.12, line_width=0,
                      annotation_text="Comfort range 22–25 °C", annotation_position="top left")
        fig.add_hrect(y0=lo, y1=SAFETY_MIN_C, fillcolor="#e63946", opacity=0.14, line_width=0)
        fig.add_hrect(y0=SAFETY_MAX_C, y1=hi, fillcolor="#e63946", opacity=0.14, line_width=0)
        fig.add_hline(y=SAFETY_MIN_C, line=dict(color="#e63946", dash="dot"),
                      annotation_text="Safety min 18 °C", annotation_position="bottom right")
        fig.add_hline(y=SAFETY_MAX_C, line=dict(color="#e63946", dash="dot"),
                      annotation_text="Safety max 30 °C", annotation_position="top right")
    fig.update_yaxes(range=[lo, hi])
    return _layout(fig, title, 430, "Indoor temperature (°C)", xtitle)


# ----------------------------------------------------------------------
# B. outdoor vs indoor
# ----------------------------------------------------------------------
def outdoor_indoor_chart(df: pd.DataFrame, floors=None, source_label: str = "") -> go.Figure:
    title = "Outdoor vs indoor temperature"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title)
    xcol, xtitle = _xcol(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[xcol], y=df["outdoor_temperature"], mode="lines", name=f"Outdoor ({source_label or 'selected source'})",
        line=dict(color=OUTDOOR_COLOR, width=3), hovertemplate="Outdoor: %{y:.1f} °C<br>%{x}<extra></extra>"))
    for n in _floors(floors):
        fig.add_trace(go.Scatter(
            x=df[xcol], y=df[f"floor{n}_temp"], mode="lines", name=f"Floor {n} indoor (simulated)",
            line=dict(color=FLOOR_COLORS[n], width=2),
            hovertemplate=f"Floor {n} indoor: %{{y:.2f}} °C<br>%{{x}}<extra></extra>"))
    return _layout(fig, title, 380, "Temperature (°C)", xtitle)


# ----------------------------------------------------------------------
# C. energy
# ----------------------------------------------------------------------
def energy_timeline_chart(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "Energy consumption per step (kWh per hour)"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title)
    xcol, xtitle = _xcol(df)
    fig = go.Figure()
    for n in _floors(floors):
        fig.add_trace(go.Bar(x=df[xcol], y=df[f"floor{n}_hvac_kwh"], name=f"Floor {n} HVAC",
                             marker_color=FLOOR_COLORS[n],
                             hovertemplate=f"Floor {n} HVAC: %{{y:.2f}} kWh<br>%{{x}}<extra></extra>"))
    fig.add_trace(go.Bar(x=df[xcol], y=df["base_energy_kwh"], name="Base building load (not floor-specific)",
                         marker_color="#6c757d", hovertemplate="Base load: %{y:.2f} kWh<br>%{x}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df[xcol], y=df["total_energy_kwh"], mode="lines+markers", name="Total building",
                             line=dict(color="#ffffff", width=2),
                             hovertemplate="Total: %{y:.2f} kWh<br>%{x}<extra></extra>"))
    fig.update_layout(barmode="stack")
    return _layout(fig, title, 380, "Energy (kWh per hourly step)", xtitle)


def floor_energy_bar(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "HVAC energy by floor (total over simulation)"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title, 300)
    chosen = _floors(floors)
    totals = [float(df[f"floor{n}_hvac_kwh"].sum()) for n in chosen]
    fig = go.Figure(go.Bar(x=[f"Floor {n}" for n in chosen], y=totals,
                           marker_color=[FLOOR_COLORS[n] for n in chosen],
                           text=[f"{t:.1f}" for t in totals], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>"))
    return _layout(fig, title, 300, "HVAC energy (kWh)")


# ----------------------------------------------------------------------
# D. HVAC action history
# ----------------------------------------------------------------------
def hvac_action_chart(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "HVAC action history (applied action after Safety Layer)"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title)
    xcol, xtitle = _xcol(df)
    chosen = _floors(floors)
    z = [df[f"floor{n}_action"].astype(int).tolist() for n in chosen]
    names = [df[f"floor{n}_action_name"].tolist() for n in chosen]
    requested = [[ACTION_NAMES[int(v)] for v in df[f"floor{n}_requested"]] for n in chosen]
    controlled = [["agent-controlled" if c else "not agent-controlled (OFF requested)"
                   for c in df[f"floor{n}_controlled"]] for n in chosen]
    custom = np.dstack([names, requested, controlled])
    scale = []
    for i, c in enumerate(ACTION_COLORS):
        scale += [[i / 4, c], [(i + 1) / 4, c]]
    fig = go.Figure(go.Heatmap(
        x=df[xcol], y=[f"Floor {n}" for n in chosen], z=z, customdata=custom, zmin=0, zmax=3,
        colorscale=scale, xgap=1, ygap=3,
        colorbar=dict(title="Action", tickvals=[0.375, 1.125, 1.875, 2.625],
                      ticktext=["OFF", "LOW", "MEDIUM", "HIGH"]),
        hovertemplate=("%{y}<br>Time: %{x}<br>Applied: <b>%{customdata[0]}</b>"
                       "<br>Requested: %{customdata[1]}<br>%{customdata[2]}<extra></extra>")))
    return _layout(fig, title, 120 + 70 * len(chosen), None, xtitle)


# ----------------------------------------------------------------------
# E. comfort
# ----------------------------------------------------------------------
def comfort_timeline_chart(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "Comfort status over time (temperature within 22–25 °C)"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title)
    xcol, xtitle = _xcol(df)
    chosen = _floors(floors)
    z = [df[f"floor{n}_in_comfort"].astype(int).tolist() for n in chosen]
    temps = [df[f"floor{n}_temp"].round(2).tolist() for n in chosen]
    fig = go.Figure(go.Heatmap(
        x=df[xcol], y=[f"Floor {n}" for n in chosen], z=z, customdata=temps, zmin=0, zmax=1,
        colorscale=[[0, "#e63946"], [0.5, "#e63946"], [0.5, "#2ec4b6"], [1, "#2ec4b6"]], xgap=1, ygap=3,
        colorbar=dict(title="Comfort", tickvals=[0.25, 0.75], ticktext=["Outside", "Inside"]),
        hovertemplate="%{y}<br>Time: %{x}<br>Temperature: %{customdata:.2f} °C<br>"
                      "In comfort range: %{z}<extra></extra>"))
    return _layout(fig, title, 120 + 70 * len(chosen), None, xtitle)


def comfort_bar_chart(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "Share of steps within the comfort range, by floor"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title, 300)
    chosen = _floors(floors)
    labels, pcts, texts, colors = [], [], [], []
    for n in chosen:
        r = comfort_ratio(df, floors=[n])
        labels.append(f"Floor {n}")
        pcts.append(r.pct)
        texts.append(f"{r.pct:.1f}%  ({r.numerator}/{r.denominator} steps)")
        colors.append(FLOOR_COLORS[n])
    fig = go.Figure(go.Bar(x=labels, y=pcts, marker_color=colors, text=texts, textposition="outside",
                           hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.update_yaxes(range=[0, 118])
    return _layout(fig, title, 300, "% of steps in 22–25 °C")


# ----------------------------------------------------------------------
# F. safety
# ----------------------------------------------------------------------
def safety_chart(df: pd.DataFrame, floors=None) -> go.Figure:
    title = "Safety analysis: physical temperature after the Safety Layer"
    if not _has_data(df):
        return empty_figure("No simulation data yet.", title)
    xcol, xtitle = _xcol(df)
    fig = go.Figure()
    for n in _floors(floors):
        pre = df[f"floor{n}_preclamp_temp"]
        cf = df[f"floor{n}_cf_temp"]
        fig.add_trace(go.Scatter(x=df[xcol], y=pre, mode="lines", name=f"Floor {n} (with Safety Layer)",
                                 line=dict(color=FLOOR_COLORS[n], width=2),
                                 hovertemplate=f"Floor {n}: %{{y:.2f}} °C<br>%{{x}}<extra></extra>"))
        bad = (pre < SAFETY_MIN_C) | (pre > SAFETY_MAX_C)
        if bad.any():
            fig.add_trace(go.Scatter(x=df.loc[bad, xcol], y=pre[bad], mode="markers",
                                     name=f"Floor {n}: violation after Safety Layer",
                                     marker=dict(color="#e63946", size=11, symbol="x"),
                                     hovertemplate=f"VIOLATION Floor {n}: %{{y:.2f}} °C<extra></extra>"))
        bad_cf = (cf < SAFETY_MIN_C) | (cf > SAFETY_MAX_C)
        if bad_cf.any():
            fig.add_trace(go.Scatter(x=df.loc[bad_cf, xcol], y=cf[bad_cf], mode="markers",
                                     name=f"Floor {n}: would violate WITHOUT Safety Layer (1-step)",
                                     marker=dict(color="#f8961e", size=9, symbol="triangle-up"),
                                     hovertemplate=f"Counterfactual Floor {n}: %{{y:.2f}} °C<extra></extra>"))
    fig.add_hline(y=SAFETY_MIN_C, line=dict(color="#e63946", dash="dot"))
    fig.add_hline(y=SAFETY_MAX_C, line=dict(color="#e63946", dash="dot"))
    fig.update_yaxes(range=[SAFETY_MIN_C - 4, SAFETY_MAX_C + 4])
    return _layout(fig, title, 400, "Temperature (°C)", xtitle)


# ----------------------------------------------------------------------
# G. reward
# ----------------------------------------------------------------------
def reward_chart(df: pd.DataFrame) -> go.Figure:
    title = "Environment reward per step and cumulative"
    if not _has_data(df) or "reward" not in df.columns:
        return empty_figure("No reward history yet.", title)
    xcol, xtitle = _xcol(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df[xcol], y=df["reward"], name="Reward per step", marker_color="#4cc9f0",
                         hovertemplate="Reward: %{y:.2f}<br>%{x}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df[xcol], y=df["reward"].cumsum(), name="Cumulative reward",
                             line=dict(color="#f9c74f", width=2.5),
                             hovertemplate="Cumulative: %{y:.1f}<br>%{x}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="Reward per step", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative reward", secondary_y=True)
    return _layout(fig, title, 360, None, xtitle)


# ----------------------------------------------------------------------
# H. cost
# ----------------------------------------------------------------------
def cost_chart(df: pd.DataFrame, tariff_label: str) -> go.Figure:
    title = f"Estimated electricity cost ({tariff_label})"
    if not _has_data(df) or "electricity_cost_inr" not in df.columns:
        return empty_figure("No cost data yet.", title)
    xcol, xtitle = _xcol(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df[xcol], y=df["electricity_cost_inr"], name="Cost per step (₹)",
                         marker_color="#90be6d", hovertemplate="₹%{y:.2f}<br>%{x}<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df[xcol], y=df["electricity_cost_inr"].cumsum(), name="Cumulative cost (₹)",
                             line=dict(color="#f9c74f", width=2.5),
                             hovertemplate="Cumulative ₹%{y:.1f}<br>%{x}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="₹ per step", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative ₹", secondary_y=True)
    return _layout(fig, title, 340, None, xtitle)


# ----------------------------------------------------------------------
# Weather and evaluation
# ----------------------------------------------------------------------
def weather_forecast_chart(hourly: pd.DataFrame, now: Optional[pd.Timestamp], source_label: str) -> go.Figure:
    title = f"Outdoor temperature and humidity - {source_label}"
    if hourly is None or len(hourly) == 0:
        return empty_figure("No hourly weather series available.", title)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=hourly["timestamp"], y=hourly["outdoor_temperature_c"], name="Temperature (°C)",
                             line=dict(color=OUTDOOR_COLOR, width=2.5),
                             hovertemplate="%{y:.1f} °C<br>%{x}<extra></extra>"), secondary_y=False)
    if "outdoor_humidity_pct" in hourly:
        fig.add_trace(go.Scatter(x=hourly["timestamp"], y=hourly["outdoor_humidity_pct"], name="Relative humidity (%)",
                                 line=dict(color="#4cc9f0", width=1.8, dash="dot"),
                                 hovertemplate="%{y:.0f} %<br>%{x}<extra></extra>"), secondary_y=True)
    if now is not None and pd.notna(now):
        fig.add_vline(x=pd.Timestamp(now).to_pydatetime(), line=dict(color="#ffffff", dash="dash"))
        fig.add_annotation(x=pd.Timestamp(now).to_pydatetime(), y=1.0, yref="paper", text="now", showarrow=False,
                           yanchor="bottom")
    fig.update_yaxes(title_text="°C", secondary_y=False)
    fig.update_yaxes(title_text="% RH", secondary_y=True)
    return _layout(fig, title, 340, None, "Local time")


CONTROLLER_NAMES = {"ppo": "PPO + Safety Layer", "baseline": "Rule-based baseline + Safety Layer"}


def evaluation_comparison_chart(summary: pd.DataFrame) -> go.Figure:
    title = "PPO vs baseline (simulated evaluation)"
    needed = {"controller", "total_energy_kwh", "total_cost_inr", "comfort_pct", "safety_pct"}
    if summary is None or summary.empty or not needed.issubset(summary.columns):
        return empty_figure("No evaluation summary available.", title)
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Energy (kWh)", "Est. cost (₹)", "Comfort (%)", "Safety (%)"))
    for i, (col, _) in enumerate([("total_energy_kwh", ""), ("total_cost_inr", ""), ("comfort_pct", ""), ("safety_pct", "")], 1):
        fig.add_trace(go.Bar(
            x=[CONTROLLER_NAMES.get(c, c) for c in summary["controller"]], y=summary[col],
            marker_color=["#4cc9f0" if c == "ppo" else "#adb5bd" for c in summary["controller"]],
            text=[("n/a" if pd.isna(v) else f"{v:.1f}") for v in summary[col]], textposition="outside",
            showlegend=False, hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=1, col=i)
    return _layout(fig, title, 360)
