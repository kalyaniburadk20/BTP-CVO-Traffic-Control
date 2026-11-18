# Machine-Learning Extension

This document describes the machine-learning components added to the BTP project:

**Control-Value-Aware Adaptive Edge Perception for Communication-Efficient Traffic Signal Control**

The ML work is an offline extension built from the existing SUMO experiment data. It does not replace the original CVO acquisition-cost logic or the network-level observation-budget planner.

## 1. Scope and objectives

The ML extension contains two distinct prediction tasks.

| Component | Prediction task | Intended use |
|---|---|---|
| Temporal traffic prediction | Predict the four directional queue lengths 10 seconds ahead | Estimate near-future intersection conditions |
| Learned-CVO estimation | Estimate whether a fresh-queue observation changes the controller's selected action | Estimate the action-divergence component of observation value |

These outputs are not interchangeable. Traffic prediction estimates future queue lengths. Learned-CVO estimates a binary action-divergence label from the fresh-queue dataset.

## 2. Temporal traffic prediction

### Dataset

- File: `results/processed/ml/traffic_prediction_dataset_temporal_v2.csv`
- Rows: 119,250
- Runs: 45
- Intersections: 5
- Prediction horizon: 10 seconds
- Evaluation: three-fold seed rotation, with each of seeds 1, 2, and 3 used once as the held-out seed

### Input features

The temporal model uses the following groups of features:

- Current observed queue lengths for north, south, east, and west.
- Current observed waiting-time values for the four directions.
- Observed signal phase and vehicle count.
- Per-second queue-change and waiting-time-change features.
- Observation age, previous observation interval, and a flag indicating whether a previous observation exists.
- Scenario, strategy, and intersection identifier as categorical features.

The model predicts four targets: `target_north_queue`, `target_south_queue`, `target_east_queue`, and `target_west_queue`.

### Model and evaluation

The implemented model is a multi-output `RandomForestRegressor` inside a scikit-learn preprocessing pipeline. Numeric features pass through the transformer; categorical features are one-hot encoded with unknown categories ignored.

The evaluation compares:

1. **Zero baseline:** predicts a queue of zero.
2. **Persistence baseline:** uses the latest observed directional queue as its prediction.
3. **Temporal Random Forest:** predicts all four future directional queues.

Reported metrics include MAE, RMSE, MAE on nonzero targets, and congestion-detection precision, recall, and F1. Congestion detection is evaluated by thresholding predicted queue values at 0.5.

### Three-fold seed-rotated results

Across the 12 scenario-direction combinations, the report records the following average reductions for Temporal RF relative to persistence:

| Metric | Reduction |
|---|---:|
| MAE | 17.61% |
| RMSE | 33.87% |
| Nonzero-target MAE | 17.16% |

The fold-level summary also shows that Temporal RF improves over persistence on MAE, RMSE, nonzero-target MAE, and F1 in all 12 scenario-direction combinations.

**Important qualification:** The zero baseline can achieve deceptively low overall MAE because many queue targets are zero. Temporal RF does not beat zero-baseline overall MAE in every scenario-direction combination. Its stronger evidence is the improvement over persistence and its better RMSE, nonzero-target MAE, and congestion-detection F1. Do not claim universal superiority over the zero baseline.

## 3. Learned-CVO estimation

### Dataset and label

- File: `results/processed/ml/learned_cvo_fresh_queue_dataset.csv`
- Rows: 26,955
- Runs: 9
- Positive-label rate: approximately 10.89%

The target is the fresh-queue action-divergence label: whether the controller's selected action using the current belief differs from the action selected using fresh queue measurements under the same signal-phase context.

The model estimates this label. It does **not** directly decide whether an observation should be acquired. The acquisition-cost comparison and network-level observation budget remain separate parts of the original CVO controller.

### Model and evaluation

The evaluated model is XGBoost with isotonic probability calibration. In each seed-rotated fold:

- One seed is used for training.
- A different seed is used for validation and calibration-threshold selection.
- The remaining seed is held out for testing.
- The roles rotate across all three seeds.

The validation-selected threshold was stable across folds:

| Held-out seed | Validation-selected threshold |
|---|---:|
| 3 | 0.22 |
| 1 | 0.21 |
| 2 | 0.22 |

### Three-fold summary

| Metric | Isotonic-calibrated XGBoost | Raw XGBoost at 0.50 |
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

The calibrated model is the preferred balanced candidate when probability quality and precision matter. Raw XGBoost has higher recall and may be relevant when avoiding missed positive cases is the primary objective. The selection is therefore application-dependent.

The always-predict-zero baseline has approximately 89.11% accuracy, but balanced accuracy is 0.50 and F1 is zero. Accuracy alone is misleading for this imbalanced dataset.

## 4. Inference interface

`src/ml/inference.py` provides reusable inference functions for the saved artifacts:

- `predict_traffic_queues(...)` returns predicted queue lengths for the four directions, using the trained temporal model.
- `predict_fresh_cvo(...)` returns the calibrated fresh-CVO probability and its threshold-based class.

A smoke test was run using existing held-out data. Both model artifacts loaded and produced finite outputs. The example CVO prediction was a false negative; this is a model error on that example, not an inference execution failure.

The learned-CVO output is an action-divergence estimate, **not a standalone acquisition command**. It must not bypass the controller's acquisition-cost and network-budget logic.

## 5. Automated tests

The inference-interface test file is `tests/test_ml_inference.py`.

The following six tests passed:

1. Traffic prediction returns four finite, nonnegative queue predictions.
2. CVO prediction returns a valid probability and binary class.
3. Missing traffic features are rejected.
4. Missing CVO features are rejected.
5. Missing values are rejected.
6. Empty input batches are rejected.

Run:

```powershell
python tests\test_ml_inference.py
```

## 6. Reproducibility and generated outputs

The following scripts and outputs form the robustness-evaluation workflow:

| File | Purpose |
|---|---|
| `src/ml/evaluate_ml_seed_robustness.py` | Runs seed-rotated offline evaluation |
| `src/ml/generate_ml_evaluation_report.py` | Generates the consolidated report, summary JSON, and plots |
| `results/processed/ml/seed_robustness/traffic_prediction_temporal_seed_cv.csv` | Fold-level temporal prediction metrics |
| `results/processed/ml/seed_robustness/traffic_prediction_temporal_seed_cv_summary.csv` | Mean and standard deviation across temporal folds |
| `results/processed/ml/seed_robustness/learned_cvo_xgboost_isotonic_seed_cv.csv` | Fold-level learned-CVO metrics |
| `results/processed/ml/seed_robustness/learned_cvo_xgboost_isotonic_seed_cv_summary.csv` | Mean and standard deviation across CVO folds |
| `results/processed/ml/seed_robustness/seed_robustness_run_manifest.json` | Evaluation configuration and run notes |
| `results/processed/ml/seed_robustness/ml_robustness_evaluation_report.md` | Consolidated evaluation report |
| `results/processed/ml/seed_robustness/ml_robustness_evaluation_summary.json` | Structured summary |
| `results/processed/ml/seed_robustness/plots/` | Generated evaluation figures |

The robustness workflow uses the existing datasets. It does not rerun the 45 original SUMO experiments.

Useful commands:

```powershell
python src\ml\evaluate_ml_seed_robustness.py
python src\ml\generate_ml_evaluation_report.py
python src\ml\inference.py
python tests\test_ml_inference.py
```

The original training scripts and model artifacts are retained under `src/ml/` and `results/processed/ml/`. Do not overwrite existing trained models or reports unintentionally. Some training scripts explicitly refuse to overwrite existing outputs.

## 7. Limitations and interpretation

- Evaluation is based on SUMO-derived data, not a deployment study using live roadside sensors or real-world traffic feeds.
- The seed-rotated evaluation provides three held-out folds. It is more informative than a single held-out seed, but it is not evidence of performance across all possible networks, cities, or traffic distributions.
- The traffic-prediction target is a 10-second queue forecast. Results should not be generalized to other forecast horizons without new evaluation.
- Queue targets are sparse and imbalanced. Report zero and persistence baselines, nonzero-target error, and detection metrics together; do not rely on accuracy or overall MAE alone.
- Learned-CVO classification has a positive-class F1 around 0.48 and recall around 0.46 for the calibrated model in the seed-rotated summary. This is not sufficient to claim a reliable standalone acquisition gate.
- Calibration and threshold selection are based on validation seeds. The held-out test seed is not used to select the threshold.
- The current inference smoke test demonstrates that the saved artifacts can be loaded and queried. It does not demonstrate live integration, real-time latency, or an end-to-end traffic-control improvement.
- The ML inference layer has not been integrated into the live SUMO control loop. The original rule-based CVO acquisition-cost and budget planner remain the operational controller.

## 8. Recommended next steps

1. Preserve the current models, datasets, reports, and original 45 SUMO results.
2. Add cost-sensitive threshold analysis for learned-CVO, explicitly weighing false negatives against false positives.
3. If the learned-CVO model is later integrated into replay or control, compare it against the original rule-based CVO and all established baselines using the same runs and metrics.
4. Measure inference latency and resource use before making edge-deployment claims.
5. Evaluate on additional seeds, traffic patterns, and ideally independent traffic data before claiming broader generalization.

## 9. Main takeaway

The strongest current ML result is the temporal Random Forest's consistent improvement over persistence across all 12 scenario-direction combinations, with aggregate reductions of 17.61% in MAE, 33.87% in RMSE, and 17.16% in nonzero-target MAE.

The learned-CVO XGBoost model provides a calibrated estimate of fresh-queue action divergence, with mean F1 of approximately 0.48 and ROC-AUC of approximately 0.85 across three seed rotations. It is a promising research extension, not yet a replacement for the original acquisition-cost and budget mechanism.
