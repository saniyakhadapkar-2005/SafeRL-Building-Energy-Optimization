# Safe RL-Based Building Energy Optimization (upgraded)

Three-floor HVAC simulation controlled by a PPO agent behind a rule-based Safety Layer, with a Streamlit dashboard,
real outdoor weather (Open-Meteo) and interactive Plotly charts.

**Honesty notes.** Indoor temperatures, energy, comfort, cost and reward are *simulated*. Outdoor weather is real only
when the dashboard says **LIVE**. Electricity cost uses a tariff *you* configure (no live pricing). No physical HVAC is
controlled. PPO results are simulation results, not measured savings.

## Windows setup (PowerShell, run inside the folder that contains `requirements.txt`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
(Smaller CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the line above.)

### Train PPO (only if `outputs\trained_model\ppo_safe_building_v3.zip` is missing or you want to retrain)
```powershell
python -m rl.train                       # 300,000 timesteps, a few minutes on CPU
python -m rl.train --timesteps 500000    # longer
```
A trained model (300k steps) is included in this ZIP. The dashboard never trains by itself; its **PPO Model** page has
a Train button that runs `rl.train` in a separate process with a progress bar.

### Evaluate and start the dashboard
```powershell
python -m rl.evaluate --episodes 10      # PPO vs baseline -> outputs\results\
python -m rl.analyze_results             # optional static PNGs
streamlit run dashboard/app.py
```
Open http://localhost:8501.

## Weather configuration
Open-Meteo needs **no API key**. Pick a city (Mumbai, Navi Mumbai, Thane, Pune, Delhi, Bengaluru, Hyderabad, Chennai,
Kolkata, Ahmedabad, Jaipur) or choose *Custom location* (search by name via Open-Meteo geocoding, or enter latitude/longitude)
on **Weather and Location**. *Refresh weather* forces a new call; otherwise readings are cached (default 10 min, adjustable).
If you later use a keyed provider, read the key from an environment variable or `.streamlit/secrets.toml`, never from source.

Fallback order when the API fails: **last successful live reading (labelled STALE)** → **simulated sample profile (labelled
SIMULATED, never shown as live)**. Weather sources for a run: Live, Simulated sample, or Historical CSV (if `data/` exists).

## Pages
Dashboard · Building Simulation · Floor Controls · Weather and Location · PPO Model · Performance Analytics · Evaluation Results · Settings.
The sidebar **Floor view** (All Floors / Floor 1 / 2 / 3) filters every page; pressing **Start** uses it as the simulation scope.
(The project's own floor names are Floor 1–3.)

## How the pieces work together
1. **Weather** (`utils/weather.py`) → hourly outdoor temperature/humidity (+ radiation) → `simulation/scenario.py` builds one row per simulated hour.
2. **Environment** (`environment/building_env.py`) consumes one row per step. Each step: observation → controller proposes 3 actions →
   **Safety Layer** (`safety/safety_layer.py`) adjusts them → physics updates each floor's temperature → energy, comfort, safety, reward.
3. **PPO** gives one *joint* 3-floor action. For a single-floor simulation only that floor's component is used; the other floors
   are not agent-controlled (they request OFF, the Safety Layer still protects them) but remain fully simulated.
   There is no separate per-floor model.
4. **SimulationSession** (`simulation/runner.py`) advances the real environment step by step; the dashboard calls it from a
   Streamlit fragment while *Running*. Pause/Resume/Reset/One-step are real session operations. Advancing happens on every page
   via a sidebar fragment; charts refresh live on the pages that show them.
5. **Charts** (`dashboard/charts.py`) plot the recorded step table. Metrics (`simulation/metrics.py`) always report numerator/denominator.

### Metric definitions
- **Comfort %** = floor-steps with temperature in 22–25 °C ÷ all floor-steps evaluated (floors shown × steps).
- **Safety % (with Safety Layer)** = floor-steps whose physical temperature after the layer's action is within 18–30 °C ÷ all floor-steps. Measured *before* the emergency clamp.
- **Safety % without layer** = one-step counterfactual: the raw agent action applied to the same state.
- Energy is kWh per hourly step; cost = grid kWh × your configured ₹/kWh.

## What changed vs the original project (audit summary)
Problems found: no `data/` folder and no model in the ZIP; the env clamped temperatures to 18–30 °C *before* checking them, so the temperature
safety flag and the safety reward penalty could never trigger; target temperature was not in the observation (PPO could not react to it);
synthetic episodes froze hour/outdoor/occupancy for the whole episode; the dashboard auto-trained PPO on the first run, used `st.line_chart`,
hard-coded constant outdoor values for the whole run, and called the filter "Floor 1/2/3" cards from final-row strings; preprocessing
sampled shuffled rows of all buildings, ignored `square_feet`, and had no humidity; `analyze_results` substituted 0 for missing metrics; no tests beyond two.
Changes: see the module docstrings. Key ones:
- **Observation v3 = 14 features** (adds three per-floor targets). The original 11-feature layout is kept as `observation_version="v2"`, so a legacy
  `ppo_safe_building_humidity_v2.zip` can still be loaded (shown as "legacy: cannot see targets"); a model with any other shape is reported incompatible, never silently used.
- Safety is recorded before the clamp; the safety reward penalty now actually fires. **The reward formula is otherwise unchanged**, but this fix changes learning signals, so models trained on the old env must be retrained.
- Synthetic episodes advance the hour with a diurnal outdoor/occupancy/solar profile.
- New: `utils/weather.py`, `simulation/` (scenario, runner, metrics), `rl/model_io.py`, multi-page dashboard, Plotly charts, tests.
- Training: reward-only normalisation, γ = 0.9, entropy 0.01, 4 parallel envs (found by comparing variants; see results below).
- `dashboard/pages/*` replaced by `dashboard/views/*` with `st.navigation`. Old model name `…humidity_v2` → `ppo_safe_building_v3`.

## Results actually obtained (sandbox, synthetic scenarios, 10 episodes × 168 h, held-out seeds)
| Controller | Total reward | Energy (kWh) | Est. cost (₹) | Comfort % | Safety % | Safety-Layer interventions |
|---|---|---|---|---|---|---|
| Rule-based baseline | 32225.7 | 11762.3 | 67969.8 | 99.3 | 100.0 | 9.8% of steps |
| PPO (300k steps) | 32315.7 | 11594.3 | 60836.5 | 99.3 | 100.0 | 25.9% of steps |

PPO is only marginally better (−1.4 % energy, −10.5 % cost, +0.3 % reward, equal comfort) and the Safety Layer intervenes on it much more often than on the baseline.
Safety was 100 % both with and without the layer in these scenarios (the simple physics rarely pushes a floor outside 18–30 °C in one step), so these runs do not demonstrate the layer preventing violations.
No dataset was available, so nothing was trained or evaluated on the ASHRAE-style CSVs; put `train.csv`, `weather_train.csv`, `building_metadata.csv` in `data/` and retrain to use them (chronological 80/20 train/eval split).

## Tests
```powershell
python -m pytest -q
```
64 tests (env, Safety Layer, per-floor scope, reset, weather failures/fallback/caching, scenario errors, metrics & denominators, empty data,
model missing/incompatible/legacy/corrupt, evaluation files, preprocessing, chart builders, headless Streamlit pages). Weather tests use mocked HTTP.
**Not verified:** the live Open-Meteo calls (the build sandbox could not reach it), the UI visually in a browser, and Windows itself (code uses `pathlib`; process launch has a Windows branch).

### Manual checklist (please run once on your PC)
1. `streamlit run dashboard/app.py` opens; header shows weather badge **LIVE** (or an explicit STALE/SIMULATED warning if offline).
2. Weather page: choose Pune → Refresh; temperature, humidity, last-update time and source update. Try Custom → search "Nagpur".
3. Floor view = Floor 2 → Start: only Floor 2 shows "agent-controlled"; Pause, Resume, One step, Reset all work; progress bar advances.
4. Floor Controls: unlink targets, set Floor 3 = 21 °C, "Apply to running simulation": Floor 3's dashed target line changes.
5. Performance Analytics: hover tooltips show time/floor/temperature/action; floor filter changes charts.
6. PPO Model: "Run inference check"; Evaluation Results: Run evaluation → table with n/N denominators.
7. Disconnect the network → Refresh weather shows a warning and SIMULATED/STALE label, never a fake LIVE badge.

## Troubleshooting
- `ModuleNotFoundError: environment` → run commands from the folder that contains `requirements.txt`.
- "PPO model not ready: expects N features" → retrain: `python -m rl.train`, or use the labelled baseline controller.
- Weather shows SIMULATED/STALE → no internet/firewall, API rate limit, or invalid coordinates; press Refresh later.
- Port busy → `streamlit run dashboard/app.py --server.port 8502`.
- Training seems stuck → check `outputs\trained_model\training.log`; a run with no heartbeat for 2 min is shown as stale and can be restarted.
- Slow install → PyTorch is large; use the CPU wheel noted above.

## Limitations
Simplified academic physics, not calibrated to a real building; Safety Layer is a rule filter, not a certified safety system;
speed/continuous operation is step-based (one hourly step per refresh tick), not real time.
