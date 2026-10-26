"""Building Simulation page: scenario settings, run controls, step log."""
import streamlit as st

from dashboard import components, state, ui
from dashboard.state import OCCUPANCY, START_TEMPS, WEATHER_SOURCES
from simulation.runner import CONTROLLER_BASELINE, CONTROLLER_LABELS, CONTROLLER_PPO, STATE_FINISHED


def render() -> None:
    components.header()
    ui.flash_messages()
    s = state.settings()
    st.subheader("Scenario")
    _, mstatus = state.get_model()

    c1, c2, c3 = st.columns(3)
    with c1:
        sources = list(WEATHER_SOURCES) if state.historical_available() else [k for k in WEATHER_SOURCES if k != "historical"]
        s["weather_source"] = st.selectbox("Outdoor weather source", sources, index=sources.index(s["weather_source"]) if s["weather_source"] in sources else 0,
                                           format_func=lambda k: WEATHER_SOURCES[k])
        if not state.historical_available():
            st.caption("Historical CSV option appears when train.csv, weather_train.csv and building_metadata.csv are in data/.")
        s["steps"] = int(st.number_input("Simulation duration (hourly steps)", 6, 168, int(s["steps"]), 6,
                                         help="24 = one day, 168 = one week. One step = one simulated hour."))
        s["start_condition"] = st.select_slider("Starting indoor condition", list(START_TEMPS), value=s["start_condition"],
                                                help="Initial temperature of every floor: " + ", ".join(f"{k} {v}°C" for k, v in START_TEMPS.items()))
    with c2:
        s["occupancy"] = st.selectbox("Occupancy scenario (assumption)", list(OCCUPANCY), index=list(OCCUPANCY).index(s["occupancy"]),
                                      format_func=lambda k: f"{k} (peak {OCCUPANCY[k]:.0f})")
        s["tariff"] = float(st.number_input("Electricity tariff (₹/kWh, user-configured)", 0.0, 50.0, float(s["tariff"]), 0.5,
                                            help="Flat price you configure. No live electricity price is fetched."))
        s["area"] = float(st.number_input("Building area (m²)", 100.0, 100000.0, float(s["area"]), 100.0))
    with c3:
        s["pv_kw"] = float(st.number_input("On-site PV capacity (kW, 0 = none)", 0.0, 500.0, float(s["pv_kw"]), 5.0,
                                           help="Solar output = capacity × 0.75 × shortwave radiation ÷ 1000 (needs a weather source with radiation)."))
        options = [CONTROLLER_PPO, CONTROLLER_BASELINE]
        s["controller"] = st.radio("Controller", options, index=options.index(s["controller"]),
                                   format_func=lambda k: CONTROLLER_LABELS[k])
        if s["controller"] == CONTROLLER_PPO and not mstatus.ready:
            st.error(f"PPO is not available: {mstatus.message}  Use the PPO Model page to train, "
                     "or switch to the clearly-labelled baseline controller.")
        if s["controller"] == CONTROLLER_BASELINE:
            st.info("Baseline mode: a simple thermostat rule drives the HVAC. This is NOT the RL agent. "
                    "The Safety Layer still applies.")
    scope = state.floor_view()
    st.caption(f"Simulation scope follows the sidebar floor selector: **{scope}**. "
               + ("All three floors are controlled by the chosen controller." if scope == "All Floors" else
                  f"Only {scope} takes the controller's action; the other floors are not agent-controlled "
                  "(they request OFF; the Safety Layer still protects them) but are fully simulated."))

    st.subheader("Run")
    components.sim_controls("sim")
    s["speed"] = int(st.select_slider("Live speed (environment steps per refresh)", [1, 2, 4, 8], value=int(s["speed"])))

    sim = state.get_sim()
    if sim is not None and not sim.finished and not state.is_running():
        if ui.button("⏩ Run all remaining steps now", key="sim_run_all"):
            bar = st.progress(0.0, text="Running simulation…")
            try:
                sim.start()
                while not sim.finished:
                    sim.step()
                    bar.progress(sim.current_step / sim.total_steps, text=f"Step {sim.current_step}/{sim.total_steps}")
                st.rerun()
            except Exception as exc:
                st.error(f"Simulation stopped: {type(exc).__name__}: {exc}")

    def live() -> None:
        sim = state.get_sim()
        if sim is None or sim.current_step == 0:
            return
        df = sim.to_frame()
        st.subheader("Step log")
        cols = ["step", "timestamp", "outdoor_temperature", "floor1_temp", "floor2_temp", "floor3_temp",
                "floor1_action_name", "floor2_action_name", "floor3_action_name", "total_energy_kwh",
                "electricity_cost_inr", "reward", "safety_override", "safety_reason"]
        ui.show_table(df[cols].tail(24).round(2))
        ui.download_button("⬇ Download simulation steps (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "simulation_steps.csv", "text/csv", key="sim_dl")

    state.run_live(live)
