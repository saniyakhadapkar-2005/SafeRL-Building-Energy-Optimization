"""Streamlit entry point:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Safe RL Building Energy", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

from dashboard import state, ui
from dashboard.views import (analytics, building_simulation, dashboard, evaluation, floor_controls, ppo_model,
                             settings_page, weather_page)

ui.inject_css()
state.init_state()

pages = [
    st.Page(dashboard.render, title="Dashboard", icon="🏢", url_path="dashboard", default=True),
    st.Page(building_simulation.render, title="Building Simulation", icon="🧪", url_path="building-simulation"),
    st.Page(floor_controls.render, title="Floor Controls", icon="🎛️", url_path="floor-controls"),
    st.Page(weather_page.render, title="Weather and Location", icon="🌦️", url_path="weather"),
    st.Page(ppo_model.render, title="PPO Model", icon="🤖", url_path="ppo-model"),
    st.Page(analytics.render, title="Performance Analytics", icon="📈", url_path="analytics"),
    st.Page(evaluation.render, title="Evaluation Results", icon="✅", url_path="evaluation"),
    st.Page(settings_page.render, title="Settings", icon="⚙️", url_path="settings"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.radio("Floor view", state.FLOOR_VIEWS, key="floor_view",
             help="Which floor(s) to display. Start uses this as the simulation scope: one floor or the entire building.")

    def clock() -> None:
        state.run_clock()
        sim = state.get_sim()
        if sim is not None:
            st.caption(f"Simulation: {sim.state} · step {sim.current_step}/{sim.total_steps}")
            st.progress(sim.current_step / sim.total_steps)

    st.fragment(run_every=state.interval_if_running())(clock)()
    st.caption("Local academic prototype — simulation only, no real HVAC is controlled.")

nav.run()
