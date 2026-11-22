# Control-Value-Aware Adaptive Edge Perception for Communication-Efficient Traffic Signal Control

A SUMO-based research project investigating intelligent traffic observation acquisition using **Control Value of Observation (CVO)** in a five-intersection traffic network.

The project implements and evaluates adaptive observation-acquisition strategies, analyzes their impact on traffic-control performance, and explores additional Machine Learning and Reinforcement Learning approaches.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Control Value of Observation](#3-control-value-of-observation)
4. [Observation Cost and Acquisition Decision](#4-observation-cost-and-acquisition-decision)
5. [Observation-Acquisition Strategies](#5-observation-acquisition-strategies)
6. [Network Configuration](#6-network-configuration)
7. [Original Network Experiments](#7-original-network-experiments)
8. [CVO Sensitivity Analysis](#8-cvo-sensitivity-analysis)
9. [Machine Learning Extensions](#9-machine-learning-extensions)
    - [Local Observation-Cost Measurement](#94-local-observation-cost-measurement)
    - [Learned-CVO Cost-Threshold Sensitivity](#95-learned-cvo-acquisition-cost-threshold-sensitivity)
10. [Exploratory Reinforcement Learning](#10-exploratory-reinforcement-learning)
11. [Experimental Visualizations](#11-experimental-visualizations)
12. [Repository Structure](#12-repository-structure)
13. [Configuration Files](#13-configuration-files)
14. [Prerequisites](#14-prerequisites)
15. [Verification and Testing](#15-verification-and-testing)
16. [Limitations](#16-limitations)
17. [Future Work](#17-future-work)

---

## 1. Project Overview

In a network of traffic intersections, continuously acquiring traffic observations can consume sensing, computation, and communication resources.

This project investigates whether an observation should be acquired based on its potential to change the downstream traffic-signal control decision.

The proposed approach uses **Control Value of Observation (CVO)** to compare:

- The action selected using the current traffic belief.
- The action selected using a hypothetical updated belief.

The central principle is to prioritize observations that can influence control decisions instead of acquiring every available observation.

### Main Objectives

- Develop a CVO-based observation-acquisition framework.
- Implement a limited observation budget across multiple intersections.
- Compare CVO with Always-Acquire, Periodic, State-Change, and AoI strategies.
- Evaluate acquisition efficiency and traffic-control performance.
- Analyze CVO sensitivity to decision margins and directional traffic changes.
- Explore Machine Learning models for traffic prediction and learned CVO.
- Investigate single-agent Deep Reinforcement Learning for traffic-signal control.

---

## 2. System Architecture

The project is organized into seven primary Python modules.

| Module | Responsibility |
|---|---|
| `src/perception/` | Traffic-state representation, observations and beliefs |
| `src/control/` | Signal controller, signal configuration and signal executor |
| `src/cvo/` | CVO calculation, hypothetical observations, observation costs and logging |
| `src/acquisition/` | Acquisition policies, network budget management and planning |
| `src/experiments/` | SUMO experiments, traffic logging, statistics and plotting |
| `src/ml/` | Traffic prediction, learned CVO datasets, training and calibration |
| `src/rl/` | Single-agent DQN implementation, training and evaluation |

### Traffic Network

The simulation contains five intersections:

`J1`, `J2`, `J3`, `J4`, and `J5`.

Three traffic-demand scenarios are considered:

- Low traffic demand
- Medium traffic demand
- High traffic demand

### Signal-Control Logic

The deterministic baseline controller calculates directional queue pressures:

$$
P_{NS} = Q_N + Q_S
$$

$$
P_{EW} = Q_E + Q_W
$$

The controller selects the direction with higher pressure:

- `NS_GREEN` when north-south pressure is higher.
- `EW_GREEN` when east-west pressure is higher.
- The existing signal phase resolves ties.

The signal executor enforces a minimum green duration and handles yellow and intermediate signal transitions.

---

## 3. Control Value of Observation

CVO measures whether a prospective observation can change the downstream traffic-signal control decision.

### 3.1 Current Belief and Current Action

Let $b_t$ denote the current traffic belief.

The signal controller is represented by $\pi$.

The current action is:

$$
a_t^* = \pi(b_t)
$$

### 3.2 Hypothetical Belief and Hypothetical Action

Let $o_{t+1}$ represent a possible new observation.

The hypothetical updated belief is:

$$
b'_{t+1} = U(b_t,o_{t+1})
$$

where $U$ represents the belief-update function.

The same controller evaluates the hypothetical belief:

$$
a_t' = \pi(b'_{t+1})
$$

Only the information available to the controller changes; the controller itself remains unchanged.

### 3.3 Binary Action Divergence

The initial CVO formulation uses binary action divergence:

$$
CVO_t =
\begin{cases}
1, & a_t^* \neq a_t' \\
0, & a_t^* = a_t'
\end{cases}
$$

Therefore:

| Condition | CVO |
|---|---:|
| Same controller action | 0 |
| Different controller actions | 1 |

The action space is:

$$
\mathcal{A} =
\{\text{NS\_GREEN},\text{EW\_GREEN}\}
$$

This formulation directly measures whether prospective information changes the selected signal action.

### 3.4 Decision Margin

The distance from the controller's decision boundary is calculated as:

$$
M_t = |P_{NS}-P_{EW}|
$$

A small margin indicates proximity to the decision boundary, while a large margin indicates greater separation between the directional queue pressures.

Decision margin provides contextual information about control sensitivity but is not itself the CVO value.

---

## 4. Observation Cost and Acquisition Decision

The acquisition framework considers three normalized observation-cost components:

1. Computation cost
2. Communication cost
3. Latency cost

The main network configuration assigns approximately equal weights to these components.

The total observation cost is calculated using the configured normalized cost components and weights.

The acquisition policy compares CVO with the total observation cost:

$$
\text{Decision} =
\begin{cases}
\text{ACQUIRE}, & CVO_t > C_t \\
\text{SKIP}, & CVO_t \leq C_t
\end{cases}
$$

where $C_t$ represents the configured total observation cost.

This allows the system to prioritize observations according to their potential impact on the controller's decision relative to their acquisition cost.

### Configured Cost Versus Measured Runtime Cost

The existing network experiment retains its configured normalized acquisition-cost baseline of **0.40**. The reference values used by the cost-normalization configuration are experiment parameters; they are not measurements of physical sensor energy, radio transmission delay, or real network latency. A separate local measurement campaign records software-side observation and payload costs. Its results are summarized in Section 9.4. A candidate normalized index of approximately **0.68** was derived for sensitivity analysis, but it was not adopted as the network configuration or used to replace the 0.40 baseline.

---

## 5. Observation-Acquisition Strategies

Five acquisition strategies are implemented and compared.

| Strategy | Description |
|---|---|
| Always-Acquire | Acquires every available observation |
| Periodic | Acquires observations at configured time intervals |
| State-Change | Acquires observations when the configured state-change condition is met |
| AoI | Uses Age of Information to prioritize observation freshness |
| CVO | Uses action divergence and observation cost to decide acquisition |

The network planner coordinates these strategies across the five intersections while respecting the observation budget.

---

## 6. Network Configuration

The main network experiment configuration specifies:

| Parameter | Configuration |
|---|---|
| Number of intersections | 5 |
| Intersections | J1–J5 |
| Traffic scenarios | Low, Medium, High |
| Simulation duration | 600 seconds |
| Warm-up duration | 60 seconds |
| Measurement interval | 60–600 seconds |
| Observation budget | 2 per timestep |
| Belief-age priority threshold | 10 seconds |
| Random seeds | 1, 2, 3 |
| Acquisition strategies | 5 |

### Network Observation Budget

The planner can acquire observations from at most two intersections in one timestep.

The 10-second belief-age threshold is a priority condition, not an unconditional guarantee that every stale intersection will be observed immediately. The observation budget can defer some acquisitions.

---

## 7. Original Network Experiments

The main experimental evaluation consists of:

$$
3 \times 5 \times 3 = 45
$$

SUMO simulation runs across:

- Three traffic-demand scenarios.
- Five observation-acquisition strategies.
- Three random seeds.

### 7.1 Original Experimental Results

The following table reports the mean results across the three random seeds.

| Scenario | Strategy | Acquisition % | Average Queue | Average Waiting | Throughput |
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

### 7.2 Main Experimental Finding

Compared with Always-Acquire, CVO reduces observation acquisition rates by:

| Scenario | Always Acquisition | CVO Acquisition | Relative Reduction |
|---|---:|---:|---:|
| Low | 40.00% | 32.96% | 17.59% |
| Medium | 40.00% | 34.36% | 14.11% |
| High | 40.00% | 31.59% | 21.02% |

CVO reduces the number of acquired observations by approximately **14–21% relative to Always-Acquire** across the three evaluated scenarios.

Throughput remains broadly comparable in the low- and medium-demand scenarios, while CVO records higher throughput in the high-demand scenario in these experiments.

However, CVO does not consistently improve average queue length or waiting time compared with Always-Acquire.

The primary result is therefore reduced observation acquisition with traffic-performance trade-offs reported explicitly, rather than a claim of universally superior traffic-control performance.

### 7.3 Statistical Results

For the original experiments, mean and standard deviation summaries are available in:

`results/processed/network_experiment_statistics.csv`

The original raw experiment logs and processed summaries are preserved in the repository.

---

## 8. CVO Sensitivity Analysis

Additional analyses investigate the relationship between CVO and controller decision sensitivity.

The analyses include:

- Decision-margin sensitivity.
- Directional perturbation sensitivity.
- Scenario-wise margin distributions.
- Action-specific sensitivity.
- Belief age and observation-budget interactions.

The original network configuration uses a hypothetical east-queue increase of 5 for its CVO perturbation.

Directional sensitivity experiments also investigate north-south and east-west perturbations separately.

An important distinction is maintained:

- CVO measures action divergence.
- Decision margin describes proximity to the decision boundary.

These are related concepts but are not interchangeable quantities.

### Important Analysis Files

- `results/processed/cvo_margin_analysis.csv`
- `results/processed/cvo_margin_analysis_by_scenario.csv`
- `results/processed/cvo_margin_by_action.csv`
- `results/processed/cvo_directional_sensitivity.csv`
- `results/processed/cvo_balanced_margin_summary.csv`

---

## 9. Machine Learning Extensions

The machine-learning extension uses datasets derived from the existing SUMO experiment logs. It contains two distinct tasks: 10-second directional queue prediction and fresh-queue action-divergence estimation. These models are currently evaluated offline; they do not replace the original CVO acquisition-cost logic or network-level observation-budget planner.

The detailed methodology, commands, results, plots, and limitations are documented in [`docs/ML_EXTENSION.md`](docs/ML_EXTENSION.md).

### 9.1 Temporal Traffic Prediction

The temporal dataset contains 119,250 rows from 45 simulation runs and five intersections. The model predicts north, south, east, and west queue lengths 10 seconds ahead.

The feature set includes current directional queues and waiting times, signal phase, vehicle count, per-second queue and waiting-time changes, observation age, previous observation interval, and categorical scenario, strategy, and intersection identifiers.

A multi-output Temporal Random Forest is evaluated against zero-queue and persistence baselines. Evaluation uses three seed-rotated folds: each of seeds 1, 2, and 3 is held out once.

#### Three-fold results

| Metric | Reduction vs. Persistence |
|---|---:|
| MAE | 17.61% |
| RMSE | 33.87% |
| Nonzero-target MAE | 17.16% |

Temporal Random Forest improves over persistence on MAE, RMSE, nonzero-target MAE, and congestion-detection F1 in all 12 scenario-direction combinations.

**Interpretation:** zero prediction remains a strong overall-MAE baseline because zero-queue observations are frequent. Temporal Random Forest does not beat zero prediction on overall MAE in every combination. Its stronger evidence is the improvement over persistence and its better RMSE, nonzero-target MAE, and congestion-detection F1. Results should not be described as universal superiority over the zero baseline.

### 9.2 Learned Fresh-Queue CVO Estimation

The fresh-queue CVO dataset contains 26,955 samples, with an approximately 10.89% positive-label rate. Its label indicates whether the controller's selected action differs when actual fresh directional queues are used under the same signal-phase context.

The evaluated classifier is XGBoost with isotonic probability calibration. Three-fold seed rotation uses one seed for training, one for validation and threshold selection, and one untouched seed for testing; the roles rotate across all three seeds.

#### Three-fold test results

| Metric | Isotonic XGBoost | Raw XGBoost (threshold 0.50) |
|---|---:|---:|
| Accuracy | 0.8909 | 0.8078 |
| Balanced accuracy | 0.7032 | 0.7593 |
| Precision | 0.4989 | 0.3228 |
| Recall | 0.4631 | 0.6971 |
| F1 | **0.4802** | 0.4410 |
| ROC-AUC | 0.8478 | 0.8485 |
| Average precision | 0.4724 | 0.4956 |
| Brier score | **0.0731** | 0.1310 |
| Log loss | **0.2553** | 0.4112 |

The validation-selected threshold was stable across folds: 0.22, 0.21, and 0.22. Isotonic calibration gives better probability quality and a higher F1 in this evaluation, while raw XGBoost has higher recall. The preferred operating point depends on the relative cost of false positives and false negatives.

The always-predict-zero baseline achieves approximately 89.11% accuracy but has balanced accuracy 0.50 and F1 0. Accuracy alone is therefore misleading for this imbalanced dataset.

**Important scope:** the classifier estimates fresh-queue action divergence under the constructed dataset's same-timestep labeling rule. Its probability/class output is not a standalone observation-acquisition command. The original CVO cost comparison and network-level observation budget remain separate. The inference module has not been integrated into the live SUMO control loop.

### 9.3 Reproducibility, Inference, and Reports

New ML scripts:

- `src/ml/evaluate_ml_seed_robustness.py` — three-fold seed-rotated offline evaluation.
- `src/ml/generate_ml_evaluation_report.py` — consolidated Markdown report, JSON summary, and plots.
- `src/ml/inference.py` — reusable traffic-queue and calibrated fresh-CVO inference functions.

Run the inference smoke test and automated interface tests:

```powershell
python src\ml\inference.py
python tests\test_ml_inference.py
```

### 9.4 Local Observation-Cost Measurement

A separate local measurement campaign was conducted across low-, medium-, and high-demand scenarios with seeds 1, 2, and 3 (nine scenario-seed runs). Each run used a 60-second warm-up followed by a 60-second measurement window. The recorder captures local model-inference batch time, Python/TraCI observation-read time, JSON serialization time, and serialized payload size.

The following pooled statistics use **planner-selected observations only** and exclude bootstrap measurements at simulation time less than or equal to 60 seconds:

| Measurement | Pooled mean | Pooled P95 |
|---|---:|---:|
| Local TraCI observation-read time | 1.0109 ms | 1.9542 ms |
| JSON serialization time | 0.0240 ms | 0.0453 ms |
| Serialized payload size | 177.79 bytes | 179 bytes |

These measurements characterize local simulation and software overhead. The SUMO/TraCI setup runs locally; payload size is a communication-volume proxy. The campaign does **not** measure actual network-transfer latency, physical sensor latency, radio energy, or bandwidth consumption. The pooled table above reports the selected-only observation metrics; model-inference timing is also recorded by the measurement utility.

### 9.5 Learned-CVO Acquisition-Cost Threshold Sensitivity

A threshold sweep compared 14 normalized acquisition-cost values on the same baseline SUMO trajectory:

**0.05, 0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.68.**

The evaluation covered three scenarios and three seeds (nine runs), producing 300 learned-CVO probability observations per run and 2,700 probability observations overall. The existing **0.40** cost policy was the only policy applied to SUMO. Every other threshold was evaluated counterfactually by a separate planner using the same per-timestep baseline beliefs. Therefore, the sweep measures one-step planner-selection sensitivity; it is **not** a set of independent closed-loop traffic-performance experiments.

The table reports the mean number of CVO-driven planner selections per run and the sample standard deviation across the three seeds. Forced age-refresh selections are excluded from these counts:

| Acquisition-cost threshold | Low demand | Medium demand | High demand |
|---:|---:|---:|---:|
| 0.10 | 76.67 ± 3.06 | 81.33 ± 2.08 | 60.67 ± 5.51 |
| 0.15 | 53.67 ± 2.52 | 71.33 ± 1.53 | 46.67 ± 6.43 |
| 0.20 | 27.00 ± 2.00 | 54.67 ± 4.93 | 34.67 ± 8.96 |
| 0.22 | 10.67 ± 0.58 | 16.33 ± 0.58 | 15.00 ± 4.58 |
| 0.30 | 6.00 ± 1.00 | 10.33 ± 1.15 | 9.67 ± 4.73 |
| 0.40 (existing baseline) | 4.00 ± 1.00 | 6.00 ± 0.00 | 3.67 ± 1.53 |
| 0.50 | 0.33 ± 0.58 | 1.33 ± 0.58 | 0.00 ± 0.00 |
| 0.68 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

At a threshold of 0.68, none of the nine runs produced a CVO-driven selection; the planner's selections at that threshold were due to the separate forced age-refresh rule. Lower thresholds produced more CVO-driven selections. The result demonstrates sensitivity of acquisition decisions to the configured cost threshold, but does not establish that a lower threshold improves traffic outcomes.

The mean predicted learned-CVO probabilities across the three seeds were approximately 0.1092 (low), 0.1235 (medium), and 0.1124 (high). The model's classification threshold (selected on validation data for binary evaluation) is conceptually different from the acquisition-cost threshold used by the planner, even when numerical values coincide.

#### Reproduction commands

~~~powershell
# Compare the existing 0.40 cost with a candidate threshold on one baseline trajectory
python scripts\measure_learned_cvo_cost_sensitivity.py low --seed 1 --candidate-cost 0.68

# Run the multi-threshold sweep for a scenario and seed
python scripts\sweep_learned_cvo_cost_thresholds.py high --seed 3

# Run the focused helper tests
python tests\test_learned_cvo_cost_sensitivity.py
python tests\test_learned_cvo_cost_threshold_sweep.py
~~~

The scripts save summary and detailed CSV outputs under **results/processed/ml/cost_sensitivity/**. Both focused threshold-analysis test files passed **4/4 tests** when run. The original 45-run results, the 0.40 baseline configuration, and the saved learned-CVO model were retained unchanged during this analysis.

## 10. Exploratory Reinforcement Learning

Single-agent reinforcement-learning experiments are implemented for intersection J1.

The remaining four intersections continue using the existing pressure-based controllers.

Two DQN versions are implemented:

- **DQN V1:** Standard Deep Q-Network with experience replay and a target network.
- **DQN V2:** Double DQN with a Dueling Q-Network and duration-aware replay.

DQN V2 uses:

- A 12-dimensional observation vector.
- Two actions: `NS_GREEN` and `EW_GREEN`.
- A 10-second decision interval.
- A duration-aware reward and discount formulation.
- A separate training and evaluation seed set.

### DQN V2 Evaluation

The held-out evaluation uses seeds 501, 502 and 503 across the three traffic scenarios.

Three policies are compared:

1. DQN V2.
2. Pressure-based control with a matched 10-second decision interval.
3. Native pressure-based control with a 1-second decision interval.

#### Overall Results

| Policy | Average Queue | Average Waiting | Average Arrivals | Average Signal Switches |
|---|---:|---:|---:|---:|
| DQN V2 | 0.8103 | 8.0442 | 116.11 | 12.56 |
| Pressure, Matched 10s | 0.7385 | 7.0274 | 117.00 | 13.33 |
| Pressure, Native 1s | 0.6014 | 5.4385 | 118.78 | 14.44 |

DQN V2 does not outperform the pressure-based baselines in this evaluation.

These results are retained as exploratory research findings and are not claimed as a traffic-control improvement.

Multi-agent reinforcement learning has not been implemented and remains a possible future extension.

#### Important RL Files

- `src/rl/dqn_agent.py`
- `src/rl/dqn_agent_v2.py`
- `src/rl/sumo_single_agent_env.py`
- `src/rl/train_dqn.py`
- `src/rl/train_dqn_v2.py`
- `src/rl/evaluate_dqn.py`
- `src/rl/evaluate_dqn_v2.py`

---

## 11. Experimental Visualizations

The original experiment plots are available in `results/processed/plots/`.

### Acquisition Rate

![Acquisition Rate](results/processed/plots/acquisition_rate.png)

### Average Queue Length

![Average Queue](results/processed/plots/average_queue.png)

### Average Waiting Time

![Average Waiting](results/processed/plots/average_waiting.png)

### Throughput

![Throughput](results/processed/plots/throughput.png)

---

## 12. Repository Structure

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
│   ├── module_interfaces.md
│   └── ML_EXTENSION.md
│
├── scripts/
│   ├── measure_learned_cvo_cost_sensitivity.py
│   └── sweep_learned_cvo_cost_thresholds.py
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
│   │   ├── inference.py
│   │   ├── evaluate_ml_seed_robustness.py
│   │   └── generate_ml_evaluation_report.py
│   ├── perception/
│   └── rl/
│
├── sumo/
├── tests/
│   ├── test_learned_cvo_cost_sensitivity.py
│   └── test_learned_cvo_cost_threshold_sweep.py
├── .gitignore
└── README.md
```

---

## 13. Configuration Files

The main configuration files are:

| File | Purpose |
|---|---|
| `configs/acquisition.yaml` | Acquisition-policy parameters |
| `configs/phase7_initial_parameters.yaml` | Initial CVO parameters |
| `configs/experiments/master.yaml` | Experiment configuration |
| `configs/experiments/network_master.yaml` | Five-intersection network experiment configuration |

The acquisition configuration currently specifies:

```yaml
periodic:
  interval: 5

state_change:
  threshold: 10

aoi:
  max_age: 10
```

The network configuration includes simulation duration, warm-up, intersection list, observation budget, belief-age threshold, scenarios, strategies, random seeds, CVO perturbation and evaluation metrics.

Experiment parameters should be preserved alongside their result files to support reproducibility.

---

## 14. Prerequisites

The project requires:

- Python.
- SUMO installed and accessible through the system PATH.
- SUMO's TraCI Python interface.
- Python packages required by the selected experiment or training script.

The repository currently does not contain a pinned `requirements.txt` or a locked Python environment.

Exact dependency versions should therefore be recorded when preparing a fresh reproducible environment.

### Reproducibility Notes

The original 45 SUMO experiment outputs are already present in `results/` and should be treated as preserved experimental results.

The existing ML datasets, trained models, raw experiment logs and processed summaries are also retained in the repository.

Ordinary documentation or analysis work should not rerun or overwrite the original experiment outputs.

---

## 15. Verification and Testing

The `tests/` directory contains standalone Python test scripts.

They are executed directly using Python rather than through pytest test-function discovery.

### Examples

```powershell
python tests\test_cvo.py
python tests\test_phase6_final.py
python tests\test_controller.py
python tests\test_controller_belief.py
python tests\test_full_controller_integration.py
python tests\test_minimum_green.py
python tests\test_yellow_transition.py
python tests\test_network_cvo.py
python tests\test_network_budget.py
python tests\test_network_acquisition.py
python tests\test_network_planner.py
python tests\test_network_cvo_decision.py
python tests\test_network_state_change.py
python tests\test_network_always.py
python tests\test_network_aoi.py
python tests\test_network_periodic.py
python tests\test_configured_policies.py
python tests\test_acquisition_metrics.py
python tests\test_cvo_acquisition_metrics.py
python tests\test_cvo_cost_cycle.py
python tests\test_cvo_no_leakage.py
python tests\test_cvo_boundary.py
python tests\test_cvo_policy.py
python tests\test_cvo_sumo.py
python src\ml\inference.py
python tests\test_ml_inference.py
```

### Selected Verification Results

| Test or Verification | Result |
|---|---|
| CVO unit tests | 3/3 passed |
| Final CVO validation | Passed |
| Controller manual tests | 7/7 passed |
| Configured acquisition policies | Passed |
| CVO acquisition-policy cases | 5/5 passed |
| CVO cost-decision cycle | Passed |
| CVO no-information-leakage validation | Passed |
| CVO decision-boundary cases | 8/8 passed |
| Minimum-green behaviour | Executed successfully |
| Yellow-transition behaviour | Executed successfully |
| Network observation-budget behaviour | Verified |
| Dynamic network planner | Executed successfully |
| ML inference interface tests | 6/6 passed |
| Learned-CVO cost-sensitivity helper tests | 4/4 passed |
| Learned-CVO threshold-sweep helper tests | 4/4 passed |
| Temporal traffic robustness evaluation | 3 held-out seed folds completed |
| Learned-CVO robustness evaluation | 3 held-out seed folds completed |
| ML inference smoke test | Completed for both saved models |
| Network acquisition | Executed successfully |
| Full controller integration | Executed successfully |

Some integration scripts completed without an explicit assertion-based pass message. These are described as successful executions rather than formal test-suite passes.

### Pytest Compatibility

The existing test files are standalone scripts and are not currently structured as pytest-discoverable test functions.

A collection-only pytest run reported:

```text
no tests collected
```

This does not mean that the standalone test scripts cannot execute; it means that pytest discovery has not been configured for them.

---

## 16. Limitations

The current implementation has several limitations:

1. Binary CVO measures whether the controller action changes, but does not quantify the magnitude of the resulting control benefit.
2. The deterministic controller and hypothetical perturbation define the scope of the current CVO formulation.
3. The 10-second belief-age setting is a priority threshold and can be affected by the network observation budget.
4. Lower observation acquisition frequency does not guarantee lower queue or waiting-time metrics.
5. Traffic-prediction performance depends strongly on the prevalence of zero-queue observations.
6. Learned CVO is evaluated under the constructed dataset's same-timestep labeling assumptions.
7. The evaluated single-agent DQN policies do not outperform the pressure-based baselines.
8. Multi-agent reinforcement learning has not been implemented.
9. The project does not yet include pinned dependency versions or pytest-compatible automated test discovery.
10. The three-fold ML robustness evaluation uses the three available simulation seeds and does not establish generalization to unseen cities, networks, or traffic distributions.
11. The learned-CVO classifier has mean F1 around 0.48 and recall around 0.46 with isotonic calibration in the seed-rotated evaluation; it is not ready to serve as the sole observation-acquisition gate.
12. The ML inference module is a tested offline interface and has not been integrated into the live SUMO control loop or benchmarked for edge-device latency.
13. Local TraCI read time and JSON serialization time do not represent measured network-transfer latency, physical sensor latency, radio energy, or bandwidth use.
14. The learned-CVO threshold sweep compares planner selections counterfactually on baseline trajectories; it does not establish closed-loop traffic-performance effects for alternative thresholds.

---

## 17. Future Work

Potential future research directions include:

- Richer action-divergence and control-benefit formulations.
- Improved belief uncertainty and observation-value estimation.
- More effective coordination of observation budgets across intersections.
- More robust temporal traffic-prediction models.
- Learned observation policies evaluated on independent traffic conditions.
- Improved reinforcement-learning policies.
- Multi-agent reinforcement learning for coordinated intersections.
- Broader evaluation across different networks, demand patterns and random seeds.
- Independent closed-loop evaluation of candidate learned-CVO cost thresholds before any alternative threshold is adopted.
- Pytest-compatible automated test coverage.
- Pinned dependencies and a reproducible environment specification.

---

## Project Summary

This project implements and evaluates a CVO-based adaptive observation-acquisition framework in a five-intersection SUMO traffic network.

Its central experimental finding is that CVO-based acquisition reduces observation frequency relative to Always-Acquire by approximately **14–21%** across the three evaluated traffic scenarios, while maintaining broadly comparable throughput in the reported experiments.

Traffic prediction, learned CVO and single-agent DQN are included as additional exploratory research extensions, with their evaluation outcomes and limitations reported separately. A separate nine-run local cost-measurement campaign and nine-run counterfactual learned-CVO threshold sweep quantify software-side observation costs and acquisition-selection sensitivity; the existing 0.40 baseline remains unchanged, and no closed-loop traffic-performance claim is made for alternative thresholds.
