"""Floor Controls page: per-floor targets, live comparison."""
import pandas as pd
import streamlit as st

from dashboard import charts, components, state, ui
from environment.building_env import TARGET_MAX_C, TARGET_MIN_C
from simulation import metrics

UI_TARGET_MIN, UI_TARGET_MAX = 20.0, 26.0


def render() -> None:
    components.header()
    ui.flash_messages()
    s = state.settings()
    st.subheader("Target temperatures")
    st.caption(f"Allowed range {TARGET_MIN_C:.0f}–{TARGET_MAX_C:.0f} °C in the environment; this page offers "
               f"{UI_TARGET_MIN:.0f}–{UI_TARGET_MAX:.0f} °C. Hard safety limits (18–30 °C) can't be changed here and the Safety Layer always applies.")
    s["link_targets"] = st.toggle("Same target for all floors", value=bool(s["link_targets"]))
    if s["link_targets"]:
        t = st.slider("Target indoor temperature (all floors)", UI_TARGET_MIN, UI_TARGET_MAX, float(s["targets"][0]), 0.5, format="%.1f °C")
        s["targets"] = [t, t, t]
    else:
        cols = st.columns(3)
        s["targets"] = [cols[i].slider(f"Floor {i+1} target", UI_TARGET_MIN, UI_TARGET_MAX, float(s["targets"][i]), 0.5, format="%.1f °C",
                                       key=f"tgt_{i}") for i in range(3)]
    sim = state.get_sim()
    c1, c2 = st.columns([1, 3])
    c1.button("Apply to running simulation", on_click=state.cb_apply_targets_live, disabled=sim is None or sim.finished,
              **ui._stretch(st.button))
    c2.caption("Otherwise the targets are used the next time you press Start after a Reset. "
               + ("Legacy v2 PPO models cannot see targets; they only influence physics and the Safety Layer."
                  if state.get_model()[1].is_legacy else ""))

    st.subheader("Floor comparison")
    components.sim_controls("floors")

    def live() -> None:
        sim = state.get_sim()
        df = sim.to_frame() if sim else None
        floors = [1, 2, 3]
        selected = state.selected_floors()
        components.floor_cards(df)
        if df is None or df.empty:
            st.info("Start a simulation to compare floors.")
            return
        rows = []
        for n in floors:
            c, sf = metrics.comfort_ratio(df, [n]), metrics.safety_ratio(df, [n])
            last = df.iloc[-1]
            rows.append({"Floor": f"Floor {n}", "Temperature (°C)": last[f"floor{n}_temp"], "Target (°C)": last[f"floor{n}_target"],
                         "Δ vs target (°C)": last[f"floor{n}_temp"] - last[f"floor{n}_target"],
                         "Applied action": last[f"floor{n}_action_name"], "HVAC energy (kWh)": df[f"floor{n}_hvac_kwh"].sum(),
                         "Comfort": c.describe(), "Safety (after layer)": sf.describe(),
                         "Agent-controlled": "yes" if last[f"floor{n}_controlled"] else "no"})
        table = pd.DataFrame(rows)
        if len(selected) == 1:
            table = table[table["Floor"] == f"Floor {selected[0]}"]
        ui.show_table(table.round(2))
        ui.show_chart(charts.temperature_chart(df, selected), "floor_temp")
        a, b = st.columns(2)
        with a:
            ui.show_chart(charts.floor_energy_bar(df, selected), "floor_energy")
        with b:
            ui.show_chart(charts.comfort_bar_chart(df, selected), "floor_comfort")

    state.run_live(live)
