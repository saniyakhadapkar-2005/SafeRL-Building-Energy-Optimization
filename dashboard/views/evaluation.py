"""Evaluation Results page: reads files produced by rl.evaluate."""
import json

import pandas as pd
import streamlit as st

from dashboard import charts, components, state, ui
from rl.model_io import RESULTS_DIR
from rl.evaluate import META_FILE, STEPS_FILE, SUMMARY_FILE, evaluate


def _read(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return None if df.empty else df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def render() -> None:
    components.header()
    ui.flash_messages()
    _, status = state.get_model()
    c1, c2 = st.columns([1, 3])
    episodes = int(c2.number_input("Episodes (each 168 hourly steps)", 1, 20, 5))
    if ui.button("Run evaluation", key="run_eval", type="primary", disabled=not status.ready):
        with st.spinner("Evaluating PPO and the baseline on identical scenarios…"):
            try:
                evaluate(episodes=episodes)
                st.success("Evaluation finished and files regenerated.")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    if not status.ready:
        st.warning(f"Evaluation needs a compatible PPO model. {status.message}")

    summary = _read(SUMMARY_FILE)
    if summary is None:
        st.info(f"No evaluation summary available ({RESULTS_DIR / SUMMARY_FILE} is missing or empty). "
                "Press Run evaluation or run `python -m rl.evaluate`.")
        return
    meta_path = RESULTS_DIR / META_FILE
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            st.caption(f"Generated {m.get('generated_at', '?')} · {m.get('episodes', '?')} episodes × {m.get('steps_per_episode', '?')} steps · "
                       f"scenarios: {m.get('scenario_source', '?')} · model: {m.get('model', '?')}")
        except (OSError, ValueError):
            pass
    st.warning("These are SIMULATED-environment results, not measured building data or guaranteed savings.")
    ui.show_chart(charts.evaluation_comparison_chart(summary), "eval_cmp")
    show = summary.copy()
    show["controller"] = show["controller"].map(charts.CONTROLLER_NAMES).fillna(show["controller"])
    for lbl, num, den in (("comfort", "comfort_num", "comfort_den"), ("safety", "safety_num", "safety_den")):
        if {num, den}.issubset(show.columns):
            show[f"{lbl} (n/N floor-steps)"] = show[num].astype(int).astype(str) + "/" + show[den].astype(int).astype(str)
    cols = [c for c in ["controller", "episodes", "steps", "total_reward", "total_energy_kwh", "total_cost_inr", "comfort_pct",
                        "comfort (n/N floor-steps)", "safety_pct", "safety (n/N floor-steps)", "safety_without_layer_pct", "override_pct"] if c in show]
    ui.show_table(show[cols].round(2))
    ppo, base = summary[summary["controller"] == "ppo"], summary[summary["controller"] == "baseline"]
    if len(ppo) and len(base) and base["total_energy_kwh"].iloc[0] and pd.notna(base["total_energy_kwh"].iloc[0]):
        d = (ppo["total_energy_kwh"].iloc[0] - base["total_energy_kwh"].iloc[0]) / base["total_energy_kwh"].iloc[0] * 100
        st.metric("PPO energy vs baseline (simulated)", f"{d:+.1f}%", help="Negative = PPO used less simulated energy. Not a measured real-world saving.")
    steps = _read(STEPS_FILE)
    if steps is not None:
        ui.download_button("⬇ Download evaluation steps (CSV)", steps.to_csv(index=False).encode("utf-8"), STEPS_FILE, "text/csv", key="eval_dl")
