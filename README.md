# Safe RL-Based Building Energy Optimization

### Safe Reinforcement Learning for Energy-Efficient and Comfortable Buildings

A Python-based project that uses **Proximal Policy Optimization (PPO)** and a rule-based Safety Layer to optimize energy consumption in a simulated three-floor building while maintaining indoor temperature within defined comfort and safety limits.

The project includes a Streamlit dashboard, simulated HVAC control, real outdoor weather integration through Open-Meteo, interactive Plotly visualizations, and performance evaluation against a rule-based baseline controller.

> **Important:** This is an academic simulation prototype. Indoor temperatures, energy consumption, comfort, electricity cost, and reward are simulated. No physical HVAC equipment is controlled, and simulation results should not be interpreted as measured real-world energy savings.


## Project Overview

Buildings consume energy to maintain comfortable indoor conditions. Traditional HVAC controllers may not adapt optimally to changing weather, occupancy, temperature targets, and energy requirements.

This project explores how Reinforcement Learning can be used to improve HVAC control decisions in a simulated three-floor building.

A PPO agent proposes HVAC control actions for the floors. Before these actions are applied, a rule-based Safety Layer checks and adjusts them according to predefined safety constraints.

The building simulation then updates indoor temperatures and calculates energy consumption, comfort, safety metrics, and reward.

A Streamlit dashboard allows users to observe the simulation, configure floor targets, inspect weather information, and compare PPO performance with a rule-based baseline.

## Problem Statement

Conventional building energy management systems may consume unnecessary electricity while attempting to maintain indoor comfort.

The main challenge is to balance three objectives:

1. Reduce simulated HVAC energy consumption.
2. Maintain comfortable indoor temperatures.
3. Keep indoor temperatures within predefined safety limits.

The project investigates whether a reinforcement learning controller, combined with a Safety Layer, can improve this balance in a simulated environment.

## Project Objectives

- Develop a simulated three-floor building environment.
- Implement a PPO-based HVAC control agent.
- Apply a rule-based Safety Layer to proposed control actions.
- Simulate the influence of outdoor conditions and occupancy on the building.
- Track energy consumption, estimated electricity cost, comfort, safety, and reward.
- Integrate outdoor weather information using Open-Meteo.
- Develop an interactive Streamlit dashboard.
- Compare PPO with a rule-based baseline controller.
- Test important environment, safety, simulation, weather, and dashboard components.

## Key Features

### 1. Reinforcement Learning with PPO

Uses Proximal Policy Optimization to learn HVAC control actions through interactions with the simulated building environment.

### 2. Safety-Constrained Control

A rule-based Safety Layer checks proposed actions and adjusts them when necessary.

### 3. Three-Floor Building Simulation

Simulates indoor temperature and HVAC energy consumption across three floors.

### 4. Weather Integration

Retrieves outdoor weather information using Open-Meteo when live data is available.

When live weather retrieval fails, the application can use a previously successful reading or a simulated weather profile, with the source identified in the interface.

### 5. Interactive Dashboard

Provides dashboard pages for monitoring building conditions, managing floor controls, viewing weather, checking the PPO model, and examining performance.

### 6. Performance Analytics

Uses Plotly charts to visualize recorded simulation data and examine temperature, energy, and controller behavior.

### 7. Controller Evaluation

Compares PPO against a rule-based baseline using reward, energy consumption, estimated electricity cost, comfort, safety, and Safety Layer interventions.

### 8. Automated Testing

Includes tests for important project components, including environment behavior, safety checks, weather handling, metrics, model loading, and dashboard-related functionality.

---

## System Architecture

The system follows this general workflow:

```text
          Outdoor Weather
                 |
                 v
       Weather and Scenario
            Generation
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
       Safe HVAC Control Actions
                 |
                 v
       Three-Floor Simulation
                 |
                 v
       Metrics and Reward
                 |
                 v
        Simulation Results
                 |
                 v
     Streamlit Dashboard
       and Plotly Charts
```

The PPO controller proposes actions, while the Safety Layer checks them before they are applied to the simulation.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Reinforcement Learning | Learning HVAC control policies |
| Proximal Policy Optimization (PPO) | Reinforcement learning algorithm |
| PyTorch | Neural network computation used by the RL stack |
| Stable-Baselines3 | PPO implementation and model training |
| Gymnasium | Reinforcement learning environment interface |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts and analytics |
| Pandas | Data processing and metric analysis |
| NumPy | Numerical computations |
| Open-Meteo | Outdoor weather data and geocoding |
| Pytest | Automated testing |

The installed versions of these dependencies are specified in `requirements.txt`.

---

## How the System Works

### Step 1: Weather and Scenario Generation

The system obtains outdoor weather information or uses a simulated weather profile.

The scenario generator creates hourly inputs for the building simulation, including relevant outdoor conditions and occupancy information.

### Step 2: Building Environment

The building environment maintains the simulated state of the three floors.

At each step, the environment provides an observation containing information needed by the controller, including floor conditions and temperature targets.

### Step 3: PPO Controller

The PPO agent receives the observation and proposes a joint HVAC action for the three floors.

The action contains a control component for each floor. When simulating an individual floor, the relevant component is used as described in the environment logic.

### Step 4: Safety Layer

The Safety Layer checks the proposed actions and adjusts them according to predefined rules.

The purpose is to reduce unsafe control behavior before actions reach the simulated building.

### Step 5: Building Simulation

The environment applies the resulting actions and updates each floor's indoor temperature and energy consumption.

The simulation also calculates comfort, safety, and reward metrics.

### Step 6: Dashboard Visualization

The Streamlit application presents simulation data through metrics, tables, and interactive charts.

Users can inspect building conditions, change settings, pause or resume the simulation, and evaluate controller performance.

---

## Reinforcement Learning Approach

### Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm that learns a policy through repeated interaction with an environment.

In this project:

- **Agent:** PPO controller.
- **Environment:** Simulated three-floor building.
- **Observation:** Building conditions and relevant control targets.
- **Action:** Proposed HVAC controls for the floors.
- **Reward:** A signal used to evaluate control behavior according to the environment's reward function.
- **Policy:** The learned mapping from observations to actions.

The agent learns by collecting experience and updating its policy during training.

The project uses a versioned observation structure so that incompatible or legacy models can be identified rather than silently treated as compatible.

### Reward and Performance

The environment calculates reward alongside energy, comfort, and safety metrics.

These measurements help evaluate controller behavior. A higher reward does not automatically mean lower energy consumption or better safety, so the metrics should be examined together.

---

## Safety Layer

The Safety Layer is a rule-based component placed between the PPO controller and the building environment.

Its main responsibilities are:

- Check proposed HVAC actions.
- Apply predefined control restrictions.
- Adjust actions when required.
- Record Safety Layer interventions.
- Support evaluation of safety-related behavior.

The simulation uses an indoor temperature safety range of **18°C to 30°C** and a comfort range of **22°C to 25°C**.

These ranges are simulation parameters, not universal building standards.

The project measures safety before the environment's emergency temperature clamp so that safety violations are not hidden by the clamp.

**Limitation:** The Safety Layer is an experimental rule filter, not a certified safety system. It does not guarantee safety in real buildings.

---

## Dashboard Modules

The application provides the following pages:

| Module | Purpose |
|---|---|
| Dashboard | Overview of the current building simulation |
| Building Simulation | Run and monitor simulated building behavior |
| Floor Controls | Configure individual floor targets and controls |
| Weather and Location | Select a location and inspect outdoor weather data |
| PPO Model | Inspect model status and run model-related checks |
| Performance Analytics | Visualize simulation metrics and trends |
| Evaluation Results | Examine controller evaluation results |
| Settings | Configure available application settings |

The dashboard supports simulation controls such as starting, pausing, resuming, resetting, and advancing a simulation step.

---

## Installation

### Prerequisites

Install the following before running the project:

- Python compatible with the dependencies in `requirements.txt`
- Git
- Internet access for retrieving live weather data

### 1. Clone the Repository

```bash
git clone https://github.com/saniyakhadapkar-2005/SafeRL-Building-Energy-Optimization.git
```

### 2. Open the Project Directory

```bash
cd SafeRL-Building-Energy-Optimization
```

Run subsequent commands from the directory containing `requirements.txt`.

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch can require a substantial download. If you need a CPU-only installation, follow the appropriate installation instructions for your operating system and Python version.

---

## Running the Project

Start the Streamlit dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

Open the local address displayed in the terminal. The default is usually:

```text
http://localhost:8501
```

The dashboard can then be used to inspect the simulation, configure available controls, and view performance metrics.

### Weather Configuration

The application uses Open-Meteo for live outdoor weather retrieval.

Select a supported city or configure a custom location in the Weather and Location page.

Live weather availability depends on network access and the weather service. If live retrieval fails, check the displayed source label to determine whether the application is using a stale reading or simulated data.

---

## Training and Evaluation

### Train the PPO Agent

If a compatible trained model is unavailable, train the agent using:

```bash
python -m rl.train
```

The training module supports a configurable timestep count. For example:

```bash
python -m rl.train --timesteps 500000
```

Training time depends on the hardware and configuration.

The dashboard does not automatically train the model on every run. Model training can also be initiated through the available model page.

### Evaluate the Controllers

Run:

```bash
python -m rl.evaluate --episodes 10
```

The evaluation compares PPO and the rule-based baseline over multiple simulated episodes.

### Generate Additional Analysis

```bash
python -m rl.analyze_results
```

This command can generate static analysis plots from available evaluation results.

Check the project's output directories for the files produced by training and evaluation.

---

## Performance Results

The repository README records the following historical evaluation from **10 simulated episodes of 168 hours each**, using synthetic scenarios and held-out random seeds.

| Metric | Rule-Based Baseline | PPO |
|---|---:|---:|
| Total reward | 32,225.7 | 32,315.7 |
| Energy consumption | 11,762.3 kWh | 11,594.3 kWh |
| Estimated electricity cost | ₹67,969.8 | ₹60,836.5 |
| Comfort | 99.3% | 99.3% |
| Safety | 100.0% | 100.0% |
| Safety Layer interventions | 9.8% of steps | 25.9% of steps |

These figures are simulation results recorded in the repository, not measurements from a real building.

### Interpretation

- PPO used approximately **1.4% less simulated energy** than the baseline in this evaluation.
- The recorded estimated cost was approximately **10.5% lower** for PPO.
- Both controllers achieved the same recorded comfort percentage.
- Both controllers recorded 100% safety in these scenarios.
- The Safety Layer intervened more frequently for PPO than for the baseline.

The results show a modest improvement in simulated energy consumption. However, they do not establish real-world energy savings or demonstrate that the Safety Layer prevented temperature violations in these particular scenarios.

Electricity costs depend on the configured tariff and evaluation settings.

---

## Running Tests

From the project root, run:

```bash
python -m pytest -q
```

The repository documents tests covering environment behavior, Safety Layer logic, floor-level simulation, weather failure handling, metrics, model compatibility, evaluation outputs, chart builders, and dashboard-related functionality.

Test results depend on the current code and environment. Run the tests locally to confirm the result for your installation.

---

## Project Structure

```text
SafeRL-Building-Energy-Optimization/
│
├── .streamlit/
│   └── Streamlit configuration
│
├── assets/
│   └── Project assets
│
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   └── Dashboard views and pages
│
├── environment/
│   └── Building reinforcement learning environment
│
├── outputs/
│   └── Model and evaluation outputs
│
├── rl/
│   ├── train.py
│   ├── evaluate.py
│   ├── analyze_results.py
│   └── Model-related modules
│
├── safety/
│   └── Safety Layer implementation
│
├── simulation/
│   └── Scenario generation, simulation runner, and metrics
│
├── tests/
│   └── Automated tests
│
├── utils/
│   └── Weather and supporting utilities
│
├── .gitignore
├── conftest.py
├── README.md
└── requirements.txt
```

Some folders contain additional modules and files that are not expanded in this overview.

---

## Limitations

- The building physics model is simplified and is not calibrated against a real building.
- Indoor temperature and energy consumption are simulated.
- Reported cost is an estimate based on the configured electricity tariff.
- PPO performance depends on training, environment assumptions, and evaluation scenarios.
- The Safety Layer is a rule-based experimental component, not a certified safety mechanism.
- Live weather retrieval requires a working network connection.
- The evaluation results do not establish real-world energy savings.
- The application is a simulation prototype and does not directly control physical HVAC equipment.

---

## Future Enhancements

Potential directions for future development include:

- Evaluate the controller with verified historical building and weather datasets.
- Improve the building model using measured energy and temperature data.
- Expand evaluation across different weather conditions and occupancy patterns.
- Study safety constraints and intervention frequency in more challenging scenarios.
- Compare PPO with additional energy optimization methods.
- Investigate deployment integrations only after appropriate validation and safety review.

These are potential improvements, not claims about features already implemented.

---

## Author

**Saniya Khadapkar**

M.Sc. Artificial Intelligence Student

GitHub: [saniyakhadapkar-2005](https://github.com/saniyakhadapkar-2005)

### Project

Safe RL-Based Building Energy Optimization

**Built with Python, Reinforcement Learning, PPO, Streamlit, and Plotly.**
