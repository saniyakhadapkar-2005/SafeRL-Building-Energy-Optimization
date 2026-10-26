# Safe RL-Based Building Energy Optimization

### Safe Reinforcement Learning for Energy-Efficient Building Management

A Python-based academic project that explores **Proximal Policy Optimization (PPO)** and rule-based safety constraints for optimizing HVAC control in a simulated three-floor building.

The project includes a Streamlit dashboard, a simulated building environment, a Safety Layer, weather integration, performance analytics, trained reinforcement learning model files, and automated tests.

> **Project Type:** Academic Simulation Prototype  
> **Primary Language:** Python  
> **Reinforcement Learning Algorithm:** Proximal Policy Optimization (PPO)  
> **Dashboard:** Streamlit  
> **GitHub Repository:** [SafeRL-Building-Energy-Optimization](https://github.com/saniyakhadapkar-2005/SafeRL-Building-Energy-Optimization)

## 1. Project Overview

Building energy management is an important application of intelligent control systems. Heating, ventilation, and air conditioning (HVAC) systems must maintain suitable indoor temperatures while managing energy consumption.

Traditional control methods use predefined rules to determine HVAC behavior. Reinforcement Learning offers another approach: an agent learns control policies through interaction with an environment and feedback from a reward function.

This project implements a Safe Reinforcement Learning-based approach for a simulated three-floor building.

A PPO agent proposes HVAC control actions. A rule-based Safety Layer checks the proposed actions before they are applied to the simulated building. The simulation then updates building conditions and calculates relevant performance metrics.

A Streamlit dashboard presents the simulation, floor controls, weather information, model status, and evaluation analytics.

**Important:** This project is a simulation prototype. It does not directly control real HVAC equipment. Energy consumption, indoor temperature, comfort, and electricity costs are simulated or calculated from simulation outputs.

## 2. Problem Statement

Building energy systems must balance energy efficiency, indoor comfort, and operational safety.

An HVAC system may consume more energy than necessary when maintaining indoor conditions. However, reducing HVAC activity without considering temperature constraints can also reduce comfort or cause unsafe simulated conditions.

The problem addressed by this project is:

**How can Reinforcement Learning and a rule-based Safety Layer be used to optimize simulated HVAC control while balancing energy consumption, indoor comfort, and temperature safety?**

The project evaluates the approach using a simulated building environment and a rule-based baseline controller.

## 3. Project Objectives

The main objectives are:

- Develop a simulated environment representing a three-floor building.
- Implement a PPO-based reinforcement learning controller.
- Apply a rule-based Safety Layer to proposed HVAC actions.
- Simulate indoor temperature and energy consumption.
- Consider relevant environmental and occupancy conditions in the simulation.
- Integrate outdoor weather information through an HTTP-based weather service.
- Develop an interactive Streamlit dashboard.
- Visualize temperature, energy, reward, and cost metrics.
- Compare PPO performance against a rule-based baseline.
- Store model files and evaluation outputs.
- Test important environment, simulation, safety, weather, and dashboard components.

## 4. Key Features

### 4.1 PPO-Based HVAC Control

Uses Proximal Policy Optimization to learn a control policy for the simulated building environment.

### 4.2 Rule-Based Safety Layer

Checks proposed control actions and applies the safety rules implemented in the project.

### 4.3 Three-Floor Building Simulation

Simulates building conditions and HVAC behavior across three floors.

### 4.4 Interactive Streamlit Dashboard

Provides a web-based interface for inspecting the simulation, adjusting available settings, and viewing performance metrics.

### 4.5 Floor-Level Controls

Provides a dedicated dashboard page for configuring available floor-related control parameters.

### 4.6 Weather Integration

Uses HTTP requests to retrieve weather information from the configured weather service. Weather availability depends on network access and service responses.

### 4.7 Performance Analytics

Uses charts and evaluation data to inspect temperature, energy consumption, reward, and estimated electricity cost.

### 4.8 Controller Evaluation

Provides an evaluation workflow for comparing PPO and a rule-based baseline.

### 4.9 Trained Model Storage

Includes a saved PPO model and accompanying metadata in the project output directory.

### 4.10 Automated Tests

Includes tests for environment behavior, model loading, preprocessing, simulation, weather, and dashboard-related functionality.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical calculations |
| Pandas | Data processing and evaluation result analysis |
| Gymnasium | Reinforcement Learning environment interface |
| Stable-Baselines3 | PPO implementation and model training |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive visualizations |
| Requests | HTTP requests for weather integration |
| Matplotlib | Static charts and visualizations |
| Pytest | Automated testing |

Project dependencies are listed in `requirements.txt`.

Stable-Baselines3 uses PyTorch as part of its reinforcement learning stack. The exact installed dependency versions depend on the Python environment and package resolution.

---

## 6. System Architecture

The project is organized into separate modules for the dashboard, reinforcement learning, building environment, safety checks, simulation, utilities, and tests.

### High-Level Architecture

```text
              User
               |
               v
      Streamlit Dashboard
               |
       +-------+--------+
       |                |
       v                v
  Floor Controls    Analytics
       |                |
       +-------+--------+
               |
               v
      Building Environment
               |
               v
       State / Observation
               |
               v
        PPO Controller
               |
               v
       Proposed HVAC Actions
               |
               v
          Safety Layer
               |
               v
       Accepted / Adjusted
          HVAC Actions
               |
               v
      Building Simulation
               |
               v
      Temperature and Energy
            Metrics
               |
               v
       Evaluation and Results
```

### Main Architecture Components

**Dashboard Layer**

The `dashboard/` directory contains the Streamlit application, interface components, application state, charts, and individual dashboard views.

**Environment Layer**

The `environment/` directory contains the building reinforcement learning environment.

**Reinforcement Learning Layer**

The `rl/` directory contains the PPO training, evaluation, model I/O, and result-analysis modules.

**Safety Layer**

The `safety/` directory contains the rule-based safety component.

**Simulation Layer**

The `simulation/` directory contains the simulation runner, scenario generation, and metrics.

**Utilities Layer**

The `utils/` directory contains weather integration and preprocessing utilities.

**Testing Layer**

The `tests/` directory contains automated tests for selected project components.

---

## 7. Project Structure

The following tree reflects the project's source files and tracked output files. Generated Python cache files are omitted because they are not required to understand the application architecture.

```text
SafeRL-Building-Energy-Optimization/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── building_dashboard_hero.png
│
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   ├── components.py
│   ├── state.py
│   ├── ui.py
│   ├── __init__.py
│   │
│   └── views/
│       ├── analytics.py
│       ├── building_simulation.py
│       ├── dashboard.py
│       ├── evaluation.py
│       ├── floor_controls.py
│       ├── ppo_model.py
│       ├── settings_page.py
│       ├── weather_page.py
│       └── __init__.py
│
├── environment/
│   ├── building_env.py
│   └── __init__.py
│
├── outputs/
│   ├── results/
│   │   ├── electricity_cost_inr.png
│   │   ├── evaluation_meta.json
│   │   ├── evaluation_steps.csv
│   │   ├── evaluation_summary.csv
│   │   ├── indoor_temperature.png
│   │   ├── reward.png
│   │   ├── summary.csv
│   │   └── total_energy_kwh.png
│   │
│   └── trained_model/
│       ├── ppo_safe_building_v3.meta.json
│       └── ppo_safe_building_v3.zip
│
├── rl/
│   ├── analyze_results.py
│   ├── evaluate.py
│   ├── model_io.py
│   ├── train.py
│   └── __init__.py
│
├── safety/
│   ├── safety_layer.py
│   └── __init__.py
│
├── simulation/
│   ├── metrics.py
│   ├── runner.py
│   ├── scenario.py
│   └── __init__.py
│
├── tests/
│   ├── conftest.py
│   ├── test_dashboard_app.py
│   ├── test_env_behavior.py
│   ├── test_humidity_observation.py
│   ├── test_model_io.py
│   ├── test_original_environment.py
│   ├── test_preprocessing.py
│   ├── test_simulation.py
│   └── test_weather.py
│
├── utils/
│   ├── preprocessing.py
│   ├── weather.py
│   └── __init__.py
│
├── .gitignore
├── conftest.py
├── README.md
└── requirements.txt
```

---

## 8. How the System Works

The system follows a sequence of observation, action selection, safety checking, simulation, and metric calculation.

### Step 1: Initialize the Application

The Streamlit application loads the dashboard modules and initializes the required application state.

### Step 2: Prepare the Building Scenario

The simulation prepares the conditions required for the building environment, including the relevant environmental inputs and control targets.

### Step 3: Obtain an Observation

The environment provides an observation representing the building state. The PPO agent uses this observation to select an action.

### Step 4: Generate a Control Action

The PPO controller proposes HVAC control actions according to its learned policy.

### Step 5: Apply Safety Checks

The Safety Layer examines the proposed actions and adjusts them when its configured rules require intervention.

### Step 6: Update the Simulation

The environment applies the resulting actions and updates simulated building conditions.

### Step 7: Calculate Metrics

The simulation calculates the available metrics, such as energy consumption, indoor temperature, reward, comfort, and safety.

### Step 8: Display Results

The dashboard presents the resulting information through metrics, tables, and charts.

### Step 9: Evaluate Controller Performance

The evaluation module can compare PPO with the rule-based baseline across simulated episodes.

---

## 9. Reinforcement Learning Approach

### 9.1 What Is Reinforcement Learning?

Reinforcement Learning is a machine learning approach in which an agent learns to make decisions by interacting with an environment.

The agent observes the current state, selects an action, and receives a reward based on the outcome.

Over time, the agent attempts to learn a policy that maximizes cumulative reward.

### 9.2 PPO Algorithm

This project uses Proximal Policy Optimization (PPO), a policy-gradient reinforcement learning algorithm.

PPO updates its policy while limiting excessively large policy changes. This helps make policy optimization more stable than unrestricted policy updates.

### 9.3 PPO Components in This Project

| Component | Project Role |
|---|---|
| Agent | PPO controller |
| Environment | Simulated building |
| Observation | Building state and relevant control information |
| Action | Proposed HVAC controls |
| Reward | Feedback calculated by the environment |
| Policy | Learned mapping from observations to actions |
| Episode | A sequence of environment interactions |

### 9.4 Training Process

The general training process is:

1. Initialize the building environment.
2. Create or load the PPO model.
3. Collect observations from the environment.
4. Generate HVAC actions.
5. Apply the actions to the environment.
6. Record rewards and transitions.
7. Update the PPO policy.
8. Save the trained model and associated metadata.

The exact observation space, action space, reward formula, and training configuration are defined in the project's implementation.

---

## 10. Safety Layer

The Safety Layer is a rule-based component intended to check control actions before they are applied to the simulated building.

Its purpose is to help prevent unsuitable actions and maintain the configured simulation constraints.

### Main Responsibilities

- Check proposed HVAC actions.
- Apply the rules implemented in `safety/safety_layer.py`.
- Adjust actions when required by those rules.
- Record or support measurement of safety interventions.
- Help evaluate controller behavior.

### Simulation Temperature Ranges

The project has previously used these configured temperature ranges:

| Parameter | Temperature |
|---|---:|
| Safety lower bound | 18°C |
| Safety upper bound | 30°C |
| Comfort lower bound | 22°C |
| Comfort upper bound | 25°C |

These values describe simulation settings, not universal building regulations.

**Safety limitation:** The Safety Layer is an experimental rule-based component. It is not a certified safety controller and does not guarantee safety in a real building.

---

## 11. Dashboard Modules

The dashboard source is organized under `dashboard/` and `dashboard/views/`.

| File | Intended Responsibility |
|---|---|
| `dashboard/app.py` | Streamlit application entry point |
| `dashboard/charts.py` | Chart-related functionality |
| `dashboard/components.py` | Reusable UI components |
| `dashboard/state.py` | Application state management |
| `dashboard/ui.py` | Shared interface functionality |
| `dashboard/views/dashboard.py` | Main dashboard view |
| `dashboard/views/building_simulation.py` | Building simulation view |
| `dashboard/views/floor_controls.py` | Floor controls |
| `dashboard/views/weather_page.py` | Weather interface |
| `dashboard/views/ppo_model.py` | PPO model interface |
| `dashboard/views/analytics.py` | Analytics view |
| `dashboard/views/evaluation.py` | Evaluation view |
| `dashboard/views/settings_page.py` | Settings interface |

These descriptions are based on filenames and project organization. The detailed behavior of each page is defined by its implementation.

---

## 12. Installation and Setup

### Prerequisites

Install:

- Python compatible with the listed dependencies.
- Git.
- Internet access for live weather retrieval.

### Step 1: Clone the Repository

```bash
git clone https://github.com/saniyakhadapkar-2005/SafeRL-Building-Energy-Optimization.git
```

### Step 2: Open the Project Directory

```bash
cd SafeRL-Building-Energy-Optimization
```

Run the remaining commands from the directory containing `requirements.txt`.

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies in requirements.txt

The project declares the following dependencies:

```text
numpy>=1.24
pandas>=2.0
gymnasium>=0.29
stable-baselines3>=2.3
streamlit>=1.40
plotly>=5.20
requests>=2.28
matplotlib>=3.7
pytest>=7.4
```

---

## 13. Running the Application

From the project root, start the Streamlit application:

```bash
streamlit run dashboard/app.py
```

After the application starts, open the local URL printed in the terminal.

The default Streamlit address is usually:

```text
http://localhost:8501
```

If the command fails, verify that the virtual environment is active and that dependencies were installed successfully.

### Weather Availability

Live weather data requires a working network connection and a valid response from the configured weather service.

If weather data is unavailable, inspect the application's weather status and error handling. Do not assume that displayed weather values are live without checking the application's source indicator.

---

## 14. Training the PPO Model

The repository contains `rl/train.py`, which is responsible for model training.

The intended module-based training command is:

```bash
python -m rl.train
```

The exact supported arguments depend on the implementation in `rl/train.py`. Check the script's argument parser before supplying optional parameters.

The trained model directory currently contains:

```text
outputs/
└── trained_model/
    ├── ppo_safe_building_v3.meta.json
    └── ppo_safe_building_v3.zip
```

The ZIP file contains the saved PPO model. The metadata file accompanies the model and may be used by the project's model-loading logic to check compatibility.

A saved model does not guarantee identical results across every environment or configuration.

---

## 15. Evaluating the Controllers

The repository contains:

- `rl/evaluate.py`
- `rl/analyze_results.py`
- `simulation/runner.py`
- `simulation/metrics.py`
- `simulation/scenario.py`

The evaluation module is intended to evaluate controller behavior in the simulated environment. The analysis module supports analysis of generated results.

The following module-based command can be tried from the project root:

```bash
python -m rl.evaluate --help
```

This displays supported command-line arguments if the script implements a compatible help interface.

After checking the available arguments, run the evaluation using the supported options.

For result analysis, inspect:

```bash
python -m rl.analyze_results --help
```

Use only the arguments supported by the actual script.

### Evaluation Metrics

Relevant metrics for the project include:

- Total reward.
- Energy consumption.
- Estimated electricity cost.
- Comfort percentage.
- Safety percentage.
- Safety Layer intervention frequency.

Metrics should be interpreted together rather than using reward alone to determine whether a controller performs better.

---

## 16. Results and Performance Metrics

The repository contains evaluation outputs in `outputs/results/`.

```text
outputs/results/
├── electricity_cost_inr.png
├── evaluation_meta.json
├── evaluation_steps.csv
├── evaluation_summary.csv
├── indoor_temperature.png
├── reward.png
├── summary.csv
└── total_energy_kwh.png
```

These files provide the basis for inspecting recorded evaluation results and generated visualizations.

### Metric Interpretation

| Metric | Meaning |
|---|---|
| Energy Consumption | Simulated energy used by the controller |
| Estimated Electricity Cost | Cost calculated from simulated energy and the configured tariff |
| Indoor Temperature | Simulated indoor temperature over time |
| Comfort Percentage | Percentage calculated according to the project's comfort metric |
| Safety Percentage | Percentage calculated according to the project's safety metric |
| Reward | Cumulative or aggregated reward calculated by the evaluation implementation |
| Safety Interventions | Frequency of actions modified by the Safety Layer |

### Results Reporting

Use the actual contents of `evaluation_summary.csv`, `summary.csv`, and `evaluation_meta.json` to populate any numerical comparison table.

Do not assume that historical results remain identical after retraining or changing the evaluation configuration.

**The results are from a simulation, not a real building.** They do not independently establish real-world energy savings, financial savings, or certified safety.

---

## 17. Running Tests

The repository includes a `tests/` directory with the following test modules:

```text
tests/
├── conftest.py
├── test_dashboard_app.py
├── test_env_behavior.py
├── test_humidity_observation.py
├── test_model_io.py
├── test_original_environment.py
├── test_preprocessing.py
├── test_simulation.py
└── test_weather.py
```

Run the test suite from the project root:

```bash
python -m pytest -q
```

To run a specific test module:

```bash
python -m pytest tests/test_simulation.py -v
```

To see more detailed output:

```bash
python -m pytest -v
```

The commands above run the tests defined in the repository. Actual pass or fail status depends on the current code, dependencies, configuration, and environment.

---

## 18. Output Files

### Trained Model

```text
outputs/trained_model/ppo_safe_building_v3.zip
```

Contains the saved PPO model.

```text
outputs/trained_model/ppo_safe_building_v3.meta.json
```

Contains associated model metadata.

### Evaluation Data

```text
outputs/results/evaluation_summary.csv
outputs/results/evaluation_steps.csv
outputs/results/evaluation_meta.json
outputs/results/summary.csv
```

These files contain recorded evaluation summaries, step-level information, metadata, or summary data according to the evaluation implementation.

### Generated Visualizations

```text
outputs/results/electricity_cost_inr.png
outputs/results/indoor_temperature.png
outputs/results/reward.png
outputs/results/total_energy_kwh.png
```

These images provide visual summaries of selected simulation or evaluation metrics.

---

## 19. Limitations

- The project uses a simulated building environment.
- The simulation is not a complete, calibrated physical model of a real building.
- Simulated energy consumption may differ from real HVAC energy use.
- Estimated electricity costs depend on the configured tariff and calculated energy.
- PPO performance depends on training conditions, model compatibility, and evaluation scenarios.
- The Safety Layer is rule-based and does not constitute a certified safety guarantee.
- Live weather retrieval depends on network access and the external service.
- Evaluation results should not be interpreted as measured real-world energy or cost savings.
- The project does not directly control physical HVAC equipment

- ##Screenshorts

- 1. Dashboard
     
<img width="696" height="693" alt="dashbord_page" src="https://github.com/user-attachments/assets/8cc2b6cf-2343-456f-b7d9-70594b26a3df" />

2.Weather and location page

<img width="698" height="672" alt="Weather and loction page" src="https://github.com/user-attachments/assets/e6d7cf5b-ca6c-4dce-af07-9c85cf346df5" />

3.PPO model page

<img width="695" height="407" alt="PPO_model_page" src="https://github.com/user-attachments/assets/6bb26e10-7174-47e3-a243-f6d72dbf19b5" />

4.evaluation result page
<img width="695" height="595" alt="Evaluation_result_page" src="https://github.com/user-attachments/assets/a9607145-5ab4-4e6a-8b64-380a96fa0b09" />
