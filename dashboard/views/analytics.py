"""Performance Analytics page: charts A–H for the current simulation."""
import streamlit as st

from dashboard import charts, components, state, ui
from simulation import metrics


def render() -> None:
    components.header()
    ui.flash_messages()
    components.sim_controls("an")
    floors = state.selected_floors()

    def live() -> None:
        sim = state.get_sim()
        df = sim.to_frame() if sim else None
        meta = st.session_state.get("sim_meta") or {}
        if df is None or df.empty:
            st.info("No simulation results yet. Start a simulation to populate these charts.")
            return
        s = metrics.summarize(df[[c for c in df.columns]])
        st.caption(f"Results from the **simulated environment** ({meta.get('controller', '')}), floors shown: "
                   f"{', '.join(f'Floor {n}' for n in floors)}.")
        ui.show_chart(charts.temperature_chart(df, floors), "an_temp")
        ui.show_chart(charts.outdoor_indoor_chart(df, floors, meta.get("weather_source", "")), "an_out")
        a, b = st.columns([1.6, 1])
        with a:
            ui.show_chart(charts.energy_timeline_chart(df, floors), "an_energy")
        with b:
            ui.show_chart(charts.floor_energy_bar(df, floors), "an_fenergy")
        st.markdown(f"**Energy:** total building {s['total_energy_kwh']:.1f} kWh (HVAC {s['hvac_energy_kwh']:.1f} kWh) over {s['steps']} hourly steps.")
        ui.show_chart(charts.hvac_action_chart(df, floors), "an_hvac")
        ui.show_chart(charts.comfort_timeline_chart(df, floors), "an_ctl")
        c1, c2 = st.columns([1, 1])
        with c1:
            ui.show_chart(charts.comfort_bar_chart(df, floors), "an_cbar")
        with c2:
            cr, sr = metrics.comfort_ratio(df, floors), metrics.safety_ratio(df, floors)
            sw = metrics.safety_ratio(df, floors, without_layer=True)
            st.markdown("##### Metric definitions")
            st.markdown(f"- **Comfort:** {cr.describe()} — denominator = floor-steps evaluated (selected floors × steps).\n"
                        f"- **Safety with Safety Layer:** {sr.describe()} — physical temperature after the layer's action, *before* the emergency clamp.\n"
                        f"- **Safety without layer (1-step counterfactual):** {sw.describe()} — raw agent action applied to the same state.\n"
                        f"- **Safety Layer interventions:** {metrics.override_ratio(df).describe('steps')}.")
        ui.show_chart(charts.safety_chart(df, floors), "an_safety")
        ui.show_chart(charts.reward_chart(df), "an_reward")
        tariff = meta.get("tariff", state.settings()["tariff"])
        ui.show_chart(charts.cost_chart(df, f"flat tariff ₹{tariff:.2f}/kWh, user-configured"), "an_cost")
        st.caption("Cost is an estimate at the configured tariff, not a measured bill or saving. "
                   "Reward is the environment's own reward for each step.")
        ui.download_button("⬇ Download results (CSV)", df.to_csv(index=False).encode("utf-8"), "simulation_steps.csv", "text/csv", key="an_dl")

    state.run_live(live)
