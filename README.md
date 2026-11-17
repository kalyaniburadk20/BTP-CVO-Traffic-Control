# Control-Value-Aware Adaptive Edge Perception for Communication-Efficient Traffic Signal Control

A SUMO-based research project that investigates how traffic-signal controllers can make intelligent observation-acquisition decisions using the **Control Value of Observation (CVO)**.

The project implements a five-intersection traffic network, compares five observation-acquisition strategies, evaluates traffic and information metrics, and explores additional machine-learning and reinforcement-learning approaches.

---

## 1. Project Overview

In a network of traffic intersections, continuously acquiring traffic observations can consume sensing, computation, and communication resources.

This project investigates whether an observation should be acquired based on its potential to change the downstream traffic-signal control decision.

The proposed approach uses Control Value of Observation (CVO), comparing:

- The action selected using the current traffic belief.
- The action selected using a hypothetical updated belief.

The binary CVO formulation is:

\[
CVO_t =
\begin{cases}
1, & a_t^* \neq a_t' \\
0, & a_t^* = a_t'
\end{cases}
\]

Where:

- \(a_t^*\) represents the current controller action.
- \(a_t'\) represents the hypothetical controller action after incorporating a prospective observation.

The central principle is to prioritize observations that can change the control decision rather than acquiring every available observation.

## 2. System Architecture

The project is organized into seven Python modules.

| Module | Responsibility |
|---|---|
| `src/perception/` | Traffic-state representation, observations and beliefs |
| `src/control/` | Signal controller, signal configuration and execution |
| `src/cvo/` | CVO calculation, hypothetical observations, observation costs and logging |
| `src/acquisition/` | Acquisition policies, budget management and network planning |
| `src/experiments/` | SUMO experiments, logging, statistics and plots |
| `src/ml/` | Traffic prediction, learned CVO datasets, model training and calibration |
| `src/rl/` | Single-agent DQN training and evaluation |

### Traffic Network

The network contains five intersections:

`J1`, `J2`, `J3`, `J4` and `J5`.

Three traffic-demand scenarios are evaluated:

- Low
- Medium
- High

### Signal Control Logic

The deterministic baseline controller calculates directional queue pressures:

\[
P_{NS} = Q_N + Q_S
\]

\[
P_{EW} = Q_E + Q_W
\]

The controller selects the direction with higher queue pressure:

- `NS_GREEN` when north-south pressure is higher.
- `EW_GREEN` when east-west pressure is higher.
- The existing signal phase resolves ties.

The signal executor enforces a minimum green duration and handles yellow and intermediate signal transitions.

## 3. Control Value of Observation

The current controller action is:

\[
a_t^* = \pi(b_t)
\]

A hypothetical observation produces an updated belief:

\[
b'_{t+1} = U(b_t,o_{t+1})
\]

The same controller evaluates the hypothetical belief:

\[
a_t' = \pi(b'_{t+1})
\]

CVO measures whether the prospective observation changes the selected action.

### Observation Cost

The acquisition decision also considers normalized observation costs:

- Computation
- Communication
- Latency

The network experiment configuration uses approximately equal weights for these three components.

The acquisition policy compares CVO against total observation cost:

- CVO greater than cost: Acquire.
- CVO less than or equal to cost: Skip.

The network planner additionally applies observation prioritization, belief freshness and a limited acquisition budget.

### Decision Margin

The controller's decision margin is:

\[
M_t = |P_{NS}-P_{EW}|
\]

A small margin indicates proximity to the controller's decision boundary, whereas a large margin indicates greater separation between directional pressures.

Decision margin is analyzed separately from CVO. It provides context about decision-boundary proximity but is not itself the CVO value.

## 4. Observation-Acquisition Strategies

Five strategies are implemented and compared.

| Strategy | Description |
|---|---|
| Always-Acquire | Acquires every available observation |
| Periodic | Acquires observations at configured intervals |
| State-Change | Acquires when the configured state-change condition is met |
| AoI | Uses Age of Information to prioritize observation freshness |
| CVO | Uses action divergence and observation cost to decide acquisition |

### Network Constraints

The main network experiment configuration specifies:

- Five intersections.
- Maximum observation budget: 2 per timestep.
- Maximum belief-age priority threshold: 10 seconds.
- Simulation duration: 600 seconds.
- Warm-up duration: 60 seconds.
- Measurement period: 60–600 seconds.
- Traffic scenarios: Low, Medium and High.
- Random seeds: 1, 2 and 3.

The 10-second belief-age threshold is a priority condition, not an unconditional guarantee that every stale intersection will be observed immediately. The two-observation budget can defer some acquisitions.

## 5. Original Network Experiment

The main experiment consists of:

\[
3 \times 5 \times 3 = 45
\]

SUMO runs across three scenarios, five acquisition strategies and three random seeds.

### Original Experimental Results

| Scenario | Strategy | Acquisition % | Avg. Queue | Avg. Waiting | Throughput |
|---|---|---:|---:|---:|---:|
| Low | Always | 40.00 | 0.2135 | 2.2486 | 51.00 |
| Low | Periodic | 10.00 | 0.2456 | 2.7516 | 50.67 |
| Low | State-Change | 10.00 | 0.2443 | 2.4183 | 51.00 |
| Low | AoI | 10.00 | 0.2448 | 2.5046 | 51.33 |
| Low | CVO | 32.96 | 0.2263 | 2.5609 | 51.00 |
| Medium | Always | 40.00 | 0.3690 | 3.6841 | 101.67 |
| Medium | Periodic | 10.00 | 0.4632 | 4.6799 | 99.67 |
| Medium | State-Change | 10.00 | 0.3804 | 2.9441 | 102.33 |
| Medium | AoI | 10.00 | 0.4389 | 4.2135 | 101.67 |
| Medium | CVO | 34.36 | 0.4480 | 4.7156 | 101.67 |
| High | Always | 40.00 | 0.8137 | 8.1477 | 199.67 |
| High | Periodic | 10.00 | 1.0249 | 9.2330 | 201.67 |
| High | State-Change | 10.00 | 1.0968 | 10.6590 | 198.33 |
| High | AoI | 10.00 | 1.0243 | 10.0791 | 200.00 |
| High | CVO | 31.59 | 0.9786 | 9.4522 | 204.33 |

### Main Finding

Compared with Always-Acquire, CVO reduces observation acquisition rates by:

| Scenario | CVO Acquisition Reduction |
|---|---:|
| Low | 17.59% |
| Medium | 14.11% |
| High | 21.02% |

CVO maintains broadly comparable throughput in the low- and medium-demand scenarios and records higher throughput in the high-demand scenario in these experiments.

However, CVO does not consistently improve average queue length or waiting time relative to Always-Acquire.

The main experimental contribution is therefore a reduction in observation acquisition frequency, with traffic-performance trade-offs reported explicitly.

## 6. CVO Sensitivity Analysis

Additional experiments investigate:

- Decision-margin sensitivity.
- Directional perturbation sensitivity.
- Scenario-wise margin distributions.
- Action-specific sensitivity.
- Belief age and observation-budget interactions.

The original network configuration uses a hypothetical east-queue increase of 5 for its CVO perturbation.

Directional sensitivity analyses also investigate north-south and east-west perturbations separately.

Decision margin and action divergence are treated as separate quantities.

## 7. Machine-Learning Extensions

Machine-learning experiments use datasets derived from the existing SUMO experiment logs.

### 7.1 Traffic Prediction

The traffic-prediction pipeline includes:

- Dataset construction from network experiment logs.
- Direction-wise traffic prediction.
- Temporal features and observation-age information.
- Random Forest models.
- XGBoost models.
- Comparison against zero-prediction and persistence baselines.

The temporal Random Forest evaluation covers 12 scenario-direction combinations.

Reported findings:

- Temporal Random Forest improves MAE over persistence in all 12 combinations.
- It improves nonzero-target MAE over persistence in all 12 combinations.
- It improves F1 over persistence in all 12 combinations.
- The zero-prediction baseline achieves lower overall MAE in all 12 combinations because of the frequency of zero-queue observations.
- Temporal Random Forest improves nonzero-target MAE over the zero baseline in the evaluated combinations.

Thus, the model provides improvements over persistence and on nonzero traffic observations, but does not establish superiority on overall MAE.

### 7.2 Learned CVO

Two learned-CVO datasets are available:

- `learned_cvo_dataset.csv`
- `learned_cvo_fresh_queue_dataset.csv`

The fresh-queue extension investigates whether actual directional queue values, rather than only a fixed hypothetical perturbation, would change the controller action.

Models include:

- Random Forest
- XGBoost
- Sigmoid calibration
- Isotonic calibration

On the held-out evaluation split, calibrated fresh-queue XGBoost achieves an F1 score of approximately **0.4818** in the reported isotonic-calibration configuration.

This model estimates same-timestep action divergence under the deterministic controller and the dataset's labeling assumptions. It should not be interpreted as an independent real-world future observation-value predictor.

## 8. Exploratory Reinforcement Learning

Single-agent reinforcement-learning experiments are implemented for intersection J1.

The remaining four intersections continue using the existing pressure-based controllers.

Two DQN versions are available:

- DQN V1: Standard DQN with experience replay and target network.
- DQN V2: Double DQN with a Dueling Q-Network and duration-aware replay.

DQN V2 uses a 10-second decision interval and is evaluated against:

- Pressure-based control with a matched 10-second decision interval.
- Native pressure-based control with a 1-second decision interval.

### Held-Out Evaluation Results

The DQN V2 evaluation uses seeds 501, 502 and 503.

| Policy | Average Queue | Average Waiting | Average Arrivals |
|---|---:|---:|---:|
| DQN V2 | 0.8103 | 8.0442 | 116.11 |
| Pressure, Matched 10s | 0.7385 | 7.0274 | 117.00 |
| Pressure, Native 1s | 0.6014 | 5.4385 | 118.78 |

DQN V2 does not outperform the pressure-based baselines in this evaluation.

These experiments are retained as exploratory work and are not claimed as a traffic-control improvement. Multi-agent reinforcement learning remains a possible future extension.

## 9. Repository Structure

```text
BTP-CVO-Traffic-Control/
│
├── configs/
│   ├── acquisition.yaml
│   ├── phase7_initial_parameters.yaml
│   └── experiments/
│       ├── master.yaml
│       └── network_master.yaml
│
├── docs/
│   ├── 07_cvo_design_notes.md
│   └── module_interfaces.md
│
├── results/
│   ├── raw/
│   ├── processed/
│   │   ├── plots/
│   │   └── ml/
│   └── rl/
│
├── src/
│   ├── acquisition/
│   ├── control/
│   ├── cvo/
│   ├── experiments/
│   ├── ml/
│   ├── perception/
│   └── rl/
│
├── sumo/
├── tests/
├── .gitignore
└── README.md
```

### Important Result Files

**Original network summaries**

- `results/processed/network_experiment_summary.csv`
- `results/processed/network_experiment_statistics.csv`

**CVO sensitivity**

- `results/processed/cvo_margin_analysis.csv`
- `results/processed/cvo_directional_sensitivity.csv`
- `results/processed/cvo_balanced_margin_summary.csv`

**Original experiment plots**

- `results/processed/plots/acquisition_rate.png`
- `results/processed/plots/average_queue.png`
- `results/processed/plots/average_waiting.png`
- `results/processed/plots/throughput.png`

Machine-learning datasets, metrics and trained models are stored under:

`results/processed/ml/`

DQN models and evaluation results are stored under:

`results/rl/`

## 10. Configuration

The main acquisition-policy configuration is:

`configs/acquisition.yaml`

The main network experiment configuration is:

`configs/experiments/network_master.yaml`

The acquisition configuration currently specifies:

```yaml
periodic:
  interval: 5

state_change:
  threshold: 10

aoi:
  max_age: 10
```

The network configuration specifies simulation duration, warm-up, intersection list, observation budget, belief-age threshold, scenarios, strategies, random seeds, CVO perturbation and evaluation metrics.

Experiment parameters should be preserved alongside their result files to support reproducibility.

## 11. Prerequisites and Reproducibility

The project requires:

- Python.
- SUMO installed and accessible through the system PATH.
- SUMO's TraCI Python interface.
- The Python packages required by the selected experiment or training script.

The project currently does not include a pinned `requirements.txt` or locked Python environment. Exact dependency versions should therefore be recorded when preparing a fresh reproducible environment.

The original 45 SUMO experiment outputs are already present in `results/`. They should be treated as preserved experimental results. Ordinary documentation or analysis work should not rerun or overwrite those experiments.

The existing ML datasets and trained models are also retained in `results/processed/ml/`.

## 12. Selected Verification

The `tests/` directory contains standalone Python test scripts.

Examples of direct execution:

```powershell
python tests\test_cvo.py
python tests\test_phase6_final.py
python tests\test_controller.py
python tests\test_minimum_green.py
python tests\test_yellow_transition.py
python tests\test_network_cvo.py
python tests\test_network_budget.py
python tests\test_network_acquisition.py
python tests\test_network_planner.py
python tests\test_network_cvo_decision.py
python tests\test_cvo_cost_cycle.py
python tests\test_cvo_no_leakage.py
python tests\test_cvo_boundary.py
python tests\test_cvo_policy.py
python tests\test_cvo_sumo.py
```

Selected validation performed during project verification includes:

- CVO unit cases: 3/3 passed.
- Final CVO validation: passed.
- Controller manual cases: 7/7 passed.
- Configured acquisition policies: passed.
- CVO acquisition metrics: passed.
- CVO cost-decision cycle: passed.
- CVO no-information-leakage validation: passed.
- CVO decision-boundary cases: passed.
- CVO acquisition-policy cases: 5/5 passed.
- Minimum-green and yellow-transition behaviours were exercised successfully.
- Network observation-budget and planner behaviours were exercised successfully.

Some integration scripts completed without an explicit assertion-based pass message; those results should be described as successful executions rather than formal test-suite passes.

The existing scripts are not currently structured as pytest-discoverable `test_*` functions. A pytest collection-only run therefore reported no tests collected.

## 13. Limitations

The current implementation has several limitations:

1. Binary CVO measures whether the controller action changes, but does not quantify the magnitude of the control benefit.
2. The deterministic controller and hypothetical perturbation define the scope of the current CVO formulation.
3. The 10-second maximum-belief-age setting is a priority threshold and can be affected by the network observation budget.
4. CVO's lower observation acquisition rate does not imply universally lower queue or waiting-time metrics.
5. Traffic-prediction performance depends strongly on the prevalence of zero-queue observations.
6. Learned CVO is evaluated under the constructed dataset's same-timestep labeling assumptions.
7. The evaluated single-agent DQN policies do not outperform the pressure-based baselines.
8. Multi-agent reinforcement learning has not been implemented.

## 14. Future Work

Potential future work includes:

- Richer action-divergence and control-benefit formulations.
- Improved belief uncertainty and observation-value estimation.
- Better coordination of observation budgets across intersections.
- More robust temporal traffic-prediction models.
- Learned observation policies evaluated on independent traffic conditions.
- Improved reinforcement-learning policies.
- Multi-agent reinforcement learning for coordinated intersections.
- Broader evaluation across networks, demand patterns and random seeds.
- Pytest-compatible automated test coverage.
- Pinned dependencies and a reproducible environment specification.

---

## Project Summary

This project implements and evaluates a CVO-based adaptive observation-acquisition framework in a five-intersection SUMO traffic network.

Its central experimental finding is that CVO-based acquisition reduces observation frequency relative to Always-Acquire by approximately **14–21%** across the three evaluated traffic scenarios, while maintaining broadly comparable throughput in the reported experiments.

Traffic prediction, learned CVO and single-agent DQN are included as additional exploratory research extensions, with their limitations and evaluation outcomes reported separately.
