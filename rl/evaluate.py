"""Evaluate the trained PPO agent against the rule-based baseline.

    python -m rl.evaluate                 # 5 episodes x 168 hourly steps
    python -m rl.evaluate --episodes 10

Both controllers run through the same ``SimulationSession`` (so the Safety Layer
sits in front of both) on identical scenarios:

* if ``data/`` CSVs exist: windows of the chronological *held-out* evaluation split
* otherwise: reproducible synthetic scenarios (labelled as such)

Outputs in ``outputs/results/``:
    evaluation_steps.csv     one row per episode/controller/step
    evaluation_summary.csv   one row per controller (aggregated; denominators included)
    evaluation_meta.json     model, data mode, seeds, timestamp
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from rl.model_io import RESULTS_DIR, load_model
from simulation import metrics
from simulation.runner import CONTROLLER_BASELINE, CONTROLLER_PPO, SimConfig, SimulationSession
from simulation.scenario import ScenarioConfig, from_historical_frame, synthetic_scenario

ROOT = Path(__file__).resolve().parents[1]
STEPS_FILE = "evaluation_steps.csv"
SUMMARY_FILE = "evaluation_summary.csv"
META_FILE = "evaluation_meta.json"


def _scenarios(episodes: int, steps: int, data_dir: Path, seed: int) -> tuple[list[pd.DataFrame], str]:
    from utils.preprocessing import data_files_present, load_prepared_data, split_train_eval
    if data_files_present(data_dir):
        df = load_prepared_data(data_dir, sample_size=100000)
        _, eval_df = split_train_eval(df, 0.2)
        cfg = ScenarioConfig(steps=steps)
        usable = max(0, len(eval_df) - steps)
        offsets = [0] if usable == 0 else [int(i * usable / max(1, episodes - 1)) if episodes > 1 else 0
                                            for i in range(episodes)]
        return ([from_historical_frame(eval_df, cfg, offset=o) for o in offsets],
                f"held-out dataset split ({len(eval_df):,} rows)")
    return [synthetic_scenario(seed + i, steps) for i in range(episodes)], "synthetic scenarios (no CSV data found)"


def evaluate(episodes: int = 5, steps: int = 168, seed: int = 1000, data_dir: Path = ROOT / "data",
             out_dir: Path = RESULTS_DIR, model_path: Optional[Path] = None) -> dict:
    if episodes < 1 or steps < 6:
        raise ValueError("episodes must be >= 1 and steps >= 6.")
    model, status = load_model(model_path)
    if not status.ready:
        raise RuntimeError(status.message)
    scenarios, data_desc = _scenarios(episodes, steps, Path(data_dir), seed)

    frames = []
    for ep, scenario in enumerate(scenarios, start=1):
        for controller in (CONTROLLER_PPO, CONTROLLER_BASELINE):
            cfg = SimConfig(controller=controller, start_temps=(26.0, 26.0, 26.0), seed=seed + ep,
                            observation_version=status.observation_version)
            session = SimulationSession(scenario, cfg, model=model if controller == CONTROLLER_PPO else None)
            frame = session.run_to_end()
            frame.insert(0, "episode", ep)
            frames.append(frame)
    steps_df = pd.concat(frames, ignore_index=True)

    rows = []
    for controller, group in steps_df.groupby("controller"):
        s = metrics.summarize(group)
        s.update({"controller": controller, "episodes": int(group["episode"].nunique())})
        rows.append(s)
    summary_df = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_df.to_csv(out_dir / STEPS_FILE, index=False)
    summary_df.to_csv(out_dir / SUMMARY_FILE, index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "model": str(status.path),
        "observation_version": status.observation_version, "episodes": episodes, "steps_per_episode": steps,
        "seed": seed, "scenario_source": data_desc,
        "note": "Simulated-environment results. Values are model estimates, not measured building data.",
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"steps": steps_df, "summary": summary_df, "meta": meta}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baseline.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=168)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.episodes, args.steps, args.seed)
    except RuntimeError as exc:
        print(f"Cannot evaluate: {exc}")
        return 1
    print("Evaluation complete -", result["meta"]["scenario_source"])
    show = ["controller", "episodes", "steps", "total_reward", "total_energy_kwh", "total_cost_inr",
            "comfort_pct", "safety_pct", "safety_without_layer_pct", "override_pct"]
    print(result["summary"][show].round(2).to_string(index=False))
    print(f"Saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
