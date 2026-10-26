"""Dashboard page: header, controls, KPIs, main charts and floor cards."""
import streamlit as st

from dashboard import charts, components, state, ui


def render() -> None:
    state.ensure_weather() if state.settings()["weather_source"] == "live" else None
    components.header()
    ui.flash_messages()
    components.sim_controls("dash")

    def live() -> None:
        sim = state.get_sim()
        df = sim.to_frame() if sim else None
        floors = state.selected_floors()
        components.kpi_row(df, sim)
        left, right = st.columns([1.35, 1])
        with left:
            ui.show_chart(charts.temperature_chart(df, floors), "dash_temp")
        with right:
            ui.show_chart(charts.energy_timeline_chart(df, floors), "dash_energy")
        components.floor_cards(df)
        a, b = st.columns(2)
        with a:
            ui.show_chart(charts.hvac_action_chart(df, floors), "dash_hvac")
        with b:
            ui.show_chart(charts.comfort_bar_chart(df, floors), "dash_comfort")
        st.caption("Indoor temperatures, HVAC energy and comfort come from the simulated environment; "
                   "outdoor values come from the selected weather source. More charts: Performance Analytics.")

    state.run_live(live)
