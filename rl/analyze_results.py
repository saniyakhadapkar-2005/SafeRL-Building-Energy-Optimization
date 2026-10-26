"""Create static PNG charts and a summary CSV from ``evaluation_steps.csv``.

    python -m rl.analyze_results

Interactive versions of these charts live in the dashboard (Plotly).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl.model_io import RESULTS_DIR
from simulation import metrics

SOURCE = RESULTS_DIR / "evaluation_steps.csv"


def main() -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} not found. Run: python -m rl.evaluate")
        return 1
    df = pd.read_csv(SOURCE)
    if df.empty:
        print(f"{SOURCE} is empty. Re-run: python -m rl.evaluate")
        return 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for controller, group in df.groupby("controller"):
        s = metrics.summarize(group)
        s["controller"] = controller
        rows.append(s)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    first_ep = df[df["episode"] == df["episode"].min()] if "episode" in df else df
    for col, title, ylabel in [("indoor_temperature", "Mean indoor temperature (episode 1)", "°C"),
                               ("total_energy_kwh", "Energy per step (episode 1)", "kWh / step"),
                               ("electricity_cost_inr", "Estimated cost per step (episode 1)", "INR / step"),
                               ("reward", "Reward per step (episode 1)", "reward")]:
        if col not in first_ep:
            continue
        plt.figure(figsize=(8, 3.5))
        for controller, g in first_ep.groupby("controller"):
            plt.plot(g["step"], g[col], label=controller)
        plt.title(title); plt.xlabel("Step (1 h)"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{col}.png", dpi=140)
        plt.close()
    print(summary.round(2).to_string(index=False))
    print(f"Results saved in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
