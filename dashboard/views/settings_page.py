"""Settings page."""
import streamlit as st

from dashboard import components, state, ui
from rl.model_io import MODEL_DIR, RESULTS_DIR


def render() -> None:
    components.header()
    ui.flash_messages()
    s = state.settings()
    st.subheader("Simulation")
    s["speed"] = int(st.select_slider("Live speed (steps per refresh)", [1, 2, 4, 8], value=int(s["speed"])))
    st.subheader("Weather refresh")
    s["weather_ttl_min"] = int(st.number_input("Minimum minutes between live API calls", 1, 120, int(s["weather_ttl_min"])))
    st.subheader("Fixed safety and comfort limits")
    st.markdown("Safety **18–30 °C**, comfort **22–25 °C**, default target **23.5 °C**. These are defined in the environment and are not user-editable.")
    st.subheader("Paths")
    st.code(f"Models:  {MODEL_DIR}\nResults: {RESULTS_DIR}\nData:    {state.DATA_DIR}  ({'found' if state.historical_available() else 'not found - using synthetic training/evaluation scenarios'})")
    st.subheader("Reset")
    if ui.button("Reset all settings and simulation", key="reset_all"):
        state.cb_reset()
        state.reset_settings()
        state.clear_model_cache()
        st.session_state.weather = None
        st.rerun()
