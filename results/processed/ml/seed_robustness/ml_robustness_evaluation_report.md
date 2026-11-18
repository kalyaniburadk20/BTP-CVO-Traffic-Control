# ML Robustness Evaluation Report

## Scope and protocol

This report summarizes the existing three-fold, seed-rotated evaluation. It uses the saved evaluation CSVs and does not retrain models or rerun SUMO.

- Traffic prediction: each seed is held out once; the other two seeds are used for training.
- Traffic baselines: zero-queue prediction and persistence using the latest observed queue.
- Learned CVO: one seed is used for training, one for validation/threshold selection and isotonic calibration, and one for final testing; roles rotate across the three folds.
- All reported summary metrics are means and standard deviations across three held-out seed folds.
- Temporal macro-averages below are unweighted averages across 12 scenario-direction cells, not pooled observation-level metrics.

## Temporal traffic prediction

### Macro-average across the 12 scenario-direction cells

| Model | MAE | RMSE | Nonzero-target MAE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 0.1367 | 0.5226 | 1.7381 | 0.0000 | 0.0000 | 0.0000 |
| Persistence | 0.2022 | 0.6273 | 1.3687 | 0.2444 | 0.2424 | 0.2432 |
| TemporalRF | 0.1581 | 0.4117 | 1.1117 | 0.5402 | 0.4063 | 0.4324 |

### Temporal RF comparisons

| Metric | Versus zero baseline | Versus persistence |
|---|---:|---:|
| MAE | 2/12 cells | 12/12 cells |
| RMSE | 12/12 cells | 12/12 cells |
| NonzeroMAE | 12/12 cells | 12/12 cells |
| F1 | 12/12 cells | 12/12 cells |

Relative reduction in the per-cell mean metric versus persistence:
- MAE: 17.61%
- RMSE: 33.87%
- Nonzero-target MAE: 17.16%

Interpretation: the zero predictor can obtain a deceptively competitive overall MAE when queue targets are sparse. Report RMSE, nonzero-target MAE, and nonzero-queue detection metrics alongside MAE; do not claim that Temporal RF universally beats the zero baseline on overall MAE.

## Learned-CVO classification

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always_predict_zero | 0.8911 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.1089 | — |
| XGBoost_raw_threshold_0.50 | 0.8078 | 0.7593 | 0.3228 | 0.6971 | 0.4410 | 0.8485 | 0.4956 | 0.1310 | 0.4112 |
| XGBoost_isotonic_validation_threshold | 0.8909 | 0.7032 | 0.4989 | 0.4631 | 0.4802 | 0.8478 | 0.4724 | 0.0731 | 0.2553 |

### Fold-level threshold selection

Validation-selected isotonic thresholds across the three folds: 0.22, 0.21, 0.22.

The calibrated model trades recall for higher precision and better probability calibration than raw XGBoost at a fixed 0.50 threshold. The zero classifier's accuracy is inflated by class imbalance; its balanced accuracy is 0.50 and it detects no positive CVO cases.

### Model-selection interpretation

- Keep isotonic-calibrated XGBoost with validation-selected threshold as the primary learned-CVO candidate when a more balanced precision/recall trade-off and usable probabilities are desired.
- Raw XGBoost may be preferable when recall is weighted more heavily, but its probability calibration is weaker.
- The validation threshold was selected by the evaluation procedure; it is not yet a cost-optimal threshold for the traffic-control system. A future decision analysis should incorporate the relative cost of missed valuable observations versus unnecessary acquisitions.

## Figures

- `plots/traffic_prediction_mae_by_scenario.png`
- `plots/traffic_prediction_rmse_by_scenario.png`
- `plots/traffic_prediction_nonzero_mae_by_scenario.png`
- `plots/traffic_prediction_f1_by_scenario.png`
- `plots/learned_cvo_f1_by_seed.png`
- `plots/learned_cvo_balanced_accuracy_by_seed.png`
- `plots/learned_cvo_brier_score_by_seed.png`

## Limitations and claims to avoid

1. Three seeds provide a useful robustness check, but the standard deviations are not a substitute for broad uncertainty estimates from many independent simulation seeds.
2. Traffic prediction is evaluated offline on previously generated SUMO data; it has not yet been shown to improve closed-loop traffic performance when integrated into the controller.
3. Learned CVO estimates the same-timestep action-divergence label defined by the dataset construction. This does not establish future real-world observation value or end-to-end system improvement.
4. Do not claim universal improvement over the zero baseline on overall traffic-prediction MAE, or universal traffic-control improvement from the ML extension.

## Reproducibility

The report is generated from the four seed-robustness CSVs already present in this directory. It does not retrain models, overwrite the saved model artifacts, or rerun SUMO.
