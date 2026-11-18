from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_DIR = PROJECT_ROOT / "results" / "processed" / "ml" / "seed_robustness"
PLOTS_DIR = ROBUSTNESS_DIR / "plots"

TEMPORAL_SUMMARY = ROBUSTNESS_DIR / "traffic_prediction_temporal_seed_cv_summary.csv"
TEMPORAL_FOLDS = ROBUSTNESS_DIR / "traffic_prediction_temporal_seed_cv.csv"
CVO_SUMMARY = ROBUSTNESS_DIR / "learned_cvo_xgboost_isotonic_seed_cv_summary.csv"
CVO_FOLDS = ROBUSTNESS_DIR / "learned_cvo_xgboost_isotonic_seed_cv.csv"

REPORT_FILE = ROBUSTNESS_DIR / "ml_robustness_evaluation_report.md"
JSON_FILE = ROBUSTNESS_DIR / "ml_robustness_evaluation_summary.json"

SCENARIO_ORDER = ["low", "medium", "high"]
DIRECTION_ORDER = ["north", "south", "east", "west"]
TEMPORAL_MODELS = ["Zero", "Persistence", "TemporalRF"]
CVO_MODELS = [
    "Always_predict_zero",
    "XGBoost_raw_threshold_0.50",
    "XGBoost_isotonic_validation_threshold",
]


def require_columns(df: pd.DataFrame, columns: list[str], file_path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")


def read_input(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required evaluation file not found: {path}\n"
            "Run the existing seed-robustness evaluation first."
        )
    df = pd.read_csv(path)
    require_columns(df, required_columns, path)
    if df.empty:
        raise ValueError(f"Evaluation file is empty: {path}")
    return df


def ordered_temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = pd.Categorical(
        out["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    out["direction"] = pd.Categorical(
        out["direction"], categories=DIRECTION_ORDER, ordered=True
    )
    out["model"] = pd.Categorical(
        out["model"], categories=TEMPORAL_MODELS, ordered=True
    )
    return out.sort_values(["scenario", "direction", "model"])


def save_grouped_temporal_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    lower_is_better: bool,
) -> None:
    pivot = df.pivot_table(
        index=["scenario", "direction"],
        columns="model",
        values=f"{metric}_mean",
        aggfunc="first",
        observed=False,
    )
    pivot = pivot.reindex(
        pd.MultiIndex.from_product(
            [SCENARIO_ORDER, DIRECTION_ORDER], names=["scenario", "direction"]
        )
    )
    for model in TEMPORAL_MODELS:
        if model not in pivot.columns:
            raise ValueError(f"Temporal summary is missing model: {model}")

    ax = pivot[TEMPORAL_MODELS].plot(kind="bar", figsize=(13, 6), width=0.82)
    ax.set_title(title)
    ax.set_xlabel("Traffic scenario / direction")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(
        [f"{scenario}\n{direction}" for scenario, direction in pivot.index],
        rotation=0,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Baseline / model")
    if lower_is_better:
        ax.text(
            0.995,
            0.985,
            "Lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    else:
        ax.text(
            0.995,
            0.985,
            "Higher is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def save_cvo_metric_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    lower_is_better: bool = False,
) -> None:
    ordered = df.set_index("model").reindex(CVO_MODELS).dropna(subset=[f"{metric}_mean"])
    labels = [
        "Always predict zero",
        "Raw XGBoost (0.50)",
        "Isotonic XGBoost\n(validation threshold)",
    ]
    x = np.arange(len(ordered))
    means = ordered[f"{metric}_mean"].to_numpy(dtype=float)
    stds = ordered[f"{metric}_std"].fillna(0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x, labels[: len(ordered)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.98,
        "Error bars = standard deviation across 3 held-out seed folds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def macro_metrics(temporal: pd.DataFrame) -> dict:
    result: dict[str, dict[str, float]] = {}
    for model in TEMPORAL_MODELS:
        subset = temporal[temporal["model"].astype(str) == model]
        result[model] = {}
        for metric in ["MAE", "RMSE", "NonzeroMAE", "Precision", "Recall", "F1"]:
            column = f"{metric}_mean"
            result[model][metric] = float(subset[column].mean())
    return result


def comparison_counts(temporal: pd.DataFrame) -> dict:
    wide = temporal.pivot_table(
        index=["scenario", "direction"],
        columns="model",
        values=["MAE_mean", "RMSE_mean", "NonzeroMAE_mean", "F1_mean"],
        aggfunc="first",
        observed=False,
    )
    result = {}
    for metric in ["MAE_mean", "RMSE_mean", "NonzeroMAE_mean", "F1_mean"]:
        result[metric] = {}
        for baseline in ["Zero", "Persistence"]:
            if (metric, baseline) not in wide.columns:
                continue
            if (metric, "TemporalRF") not in wide.columns:
                continue
            rf = wide[(metric, "TemporalRF")]
            base = wide[(metric, baseline)]
            valid = rf.notna() & base.notna()
            better = (rf < base) if metric != "F1_mean" else (rf > base)
            result[metric][baseline] = {
                "better_cells": int((better & valid).sum()),
                "total_cells": int(valid.sum()),
            }
    return result


def mean_relative_reduction(temporal: pd.DataFrame, baseline: str, metric: str) -> float | None:
    wide = temporal.pivot_table(
        index=["scenario", "direction"],
        columns="model",
        values=f"{metric}_mean",
        aggfunc="first",
        observed=False,
    )
    if baseline not in wide or "TemporalRF" not in wide:
        return None
    valid = wide[baseline] > 0
    if not valid.any():
        return None
    values = (wide.loc[valid, baseline] - wide.loc[valid, "TemporalRF"]) / wide.loc[valid, baseline]
    return float(values.mean() * 100.0)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    temporal_required = [
        "scenario", "direction", "model", "folds",
        "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
        "NonzeroMAE_mean", "NonzeroMAE_std",
        "Precision_mean", "Recall_mean", "F1_mean", "F1_std",
    ]
    cvo_required = [
        "model", "folds", "accuracy_mean", "balanced_accuracy_mean",
        "precision_mean", "precision_std", "recall_mean", "recall_std",
        "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
        "average_precision_mean", "average_precision_std",
        "brier_score_mean", "brier_score_std", "log_loss_mean", "log_loss_std",
    ]

    temporal = read_input(TEMPORAL_SUMMARY, temporal_required)
    cvo = read_input(CVO_SUMMARY, cvo_required)
    temporal_folds = read_input(
        TEMPORAL_FOLDS,
        ["test_seed", "train_seeds", "model"] if "train_seeds" in pd.read_csv(TEMPORAL_FOLDS, nrows=0).columns else ["test_seed", "model"],
    )
    cvo_folds = read_input(CVO_FOLDS, ["test_seed", "train_seed", "validation_seed", "model", "threshold"])

    temporal["folds"] = pd.to_numeric(temporal["folds"], errors="raise")
    cvo["folds"] = pd.to_numeric(cvo["folds"], errors="raise")
    if not (temporal["folds"] == 3).all():
        raise ValueError("Temporal summary should report exactly three folds per row.")
    if not (cvo["folds"] == 3).all():
        raise ValueError("Learned-CVO summary should report exactly three folds per model.")

    temporal = ordered_temporal(temporal)
    if temporal[["scenario", "direction", "model"]].isna().any().any():
        raise ValueError("Temporal summary contains unexpected scenario/direction/model labels.")
    unique_cells = temporal[["scenario", "direction"]].drop_duplicates()
    if len(unique_cells) != 12:
        raise ValueError(f"Expected 12 scenario-direction cells, found {len(unique_cells)}.")
    for model in TEMPORAL_MODELS:
        if not (temporal["model"].astype(str) == model).any():
            raise ValueError(f"Temporal summary does not contain model {model}.")

    cvo = cvo.copy()
    if set(CVO_MODELS) - set(cvo["model"].astype(str)):
        raise ValueError("Learned-CVO summary is missing one or more expected model rows.")

    # Charts: one chart per metric for readability.
    save_grouped_temporal_plot(
        temporal, "MAE", "MAE (queue vehicles)",
        "Temporal queue prediction: MAE across scenario-direction cells",
        "traffic_prediction_mae_by_scenario.png", True,
    )
    save_grouped_temporal_plot(
        temporal, "RMSE", "RMSE (queue vehicles)",
        "Temporal queue prediction: RMSE across scenario-direction cells",
        "traffic_prediction_rmse_by_scenario.png", True,
    )
    save_grouped_temporal_plot(
        temporal, "NonzeroMAE", "MAE on nonzero queue targets (vehicles)",
        "Temporal queue prediction: nonzero-target MAE",
        "traffic_prediction_nonzero_mae_by_scenario.png", True,
    )
    save_grouped_temporal_plot(
        temporal, "F1", "F1-score for nonzero-queue detection",
        "Temporal queue prediction: congestion-detection F1",
        "traffic_prediction_f1_by_scenario.png", False,
    )
    save_cvo_metric_plot(
        cvo, "f1", "Learned-CVO classifier: F1 across seed folds",
        "F1-score", "learned_cvo_f1_by_seed.png", False,
    )
    save_cvo_metric_plot(
        cvo, "balanced_accuracy", "Learned-CVO classifier: balanced accuracy",
        "Balanced accuracy", "learned_cvo_balanced_accuracy_by_seed.png", False,
    )
    save_cvo_metric_plot(
        cvo, "brier_score", "Learned-CVO classifier: probability calibration",
        "Brier score (lower is better)", "learned_cvo_brier_score_by_seed.png", True,
    )

    macros = macro_metrics(temporal)
    counts = comparison_counts(temporal)
    cvo_index = cvo.set_index("model")
    calibrated_name = "XGBoost_isotonic_validation_threshold"
    raw_name = "XGBoost_raw_threshold_0.50"
    zero_name = "Always_predict_zero"

    selected_fold_thresholds = cvo_folds[
        cvo_folds["model"].astype(str) == calibrated_name
    ]["threshold"].astype(float)
    threshold_values = [float(v) for v in selected_fold_thresholds.tolist()]

    report_lines = [
        "# ML Robustness Evaluation Report",
        "",
        "## Scope and protocol",
        "",
        "This report summarizes the existing three-fold, seed-rotated evaluation. It uses the saved evaluation CSVs and does not retrain models or rerun SUMO.",
        "",
        "- Traffic prediction: each seed is held out once; the other two seeds are used for training.",
        "- Traffic baselines: zero-queue prediction and persistence using the latest observed queue.",
        "- Learned CVO: one seed is used for training, one for validation/threshold selection and isotonic calibration, and one for final testing; roles rotate across the three folds.",
        "- All reported summary metrics are means and standard deviations across three held-out seed folds.",
        "- Temporal macro-averages below are unweighted averages across 12 scenario-direction cells, not pooled observation-level metrics.",
        "",
        "## Temporal traffic prediction",
        "",
        "### Macro-average across the 12 scenario-direction cells",
        "",
        "| Model | MAE | RMSE | Nonzero-target MAE | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in TEMPORAL_MODELS:
        m = macros[model]
        report_lines.append(
            f"| {model} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['NonzeroMAE']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} |"
        )

    report_lines.extend([
        "",
        "### Temporal RF comparisons",
        "",
        "| Metric | Versus zero baseline | Versus persistence |",
        "|---|---:|---:|",
    ])
    for metric in ["MAE_mean", "RMSE_mean", "NonzeroMAE_mean", "F1_mean"]:
        label = metric.replace("_mean", "")
        z = counts[metric].get("Zero", {}).get("better_cells", 0)
        p = counts[metric].get("Persistence", {}).get("better_cells", 0)
        zt = counts[metric].get("Zero", {}).get("total_cells", 0)
        pt = counts[metric].get("Persistence", {}).get("total_cells", 0)
        report_lines.append(f"| {label} | {z}/{zt} cells | {p}/{pt} cells |")

    mae_red_persistence = mean_relative_reduction(temporal, "Persistence", "MAE")
    rmse_red_persistence = mean_relative_reduction(temporal, "Persistence", "RMSE")
    nonzero_red_persistence = mean_relative_reduction(temporal, "Persistence", "NonzeroMAE")
    report_lines.extend([
        "",
        "Relative reduction in the per-cell mean metric versus persistence:",
        f"- MAE: {mae_red_persistence:.2f}%" if mae_red_persistence is not None else "- MAE: unavailable",
        f"- RMSE: {rmse_red_persistence:.2f}%" if rmse_red_persistence is not None else "- RMSE: unavailable",
        f"- Nonzero-target MAE: {nonzero_red_persistence:.2f}%" if nonzero_red_persistence is not None else "- Nonzero-target MAE: unavailable",
        "",
        "Interpretation: the zero predictor can obtain a deceptively competitive overall MAE when queue targets are sparse. Report RMSE, nonzero-target MAE, and nonzero-queue detection metrics alongside MAE; do not claim that Temporal RF universally beats the zero baseline on overall MAE.",
        "",
        "## Learned-CVO classification",
        "",
        "| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for model in [zero_name, raw_name, calibrated_name]:
        row = cvo_index.loc[model]
        def fmt(column: str) -> str:
            value = row[column]
            return "—" if pd.isna(value) else f"{float(value):.4f}"
        report_lines.append(
            f"| {model} | {fmt('accuracy_mean')} | {fmt('balanced_accuracy_mean')} | {fmt('precision_mean')} | {fmt('recall_mean')} | {fmt('f1_mean')} | {fmt('roc_auc_mean')} | {fmt('average_precision_mean')} | {fmt('brier_score_mean')} | {fmt('log_loss_mean')} |"
        )

    report_lines.extend([
        "",
        "### Fold-level threshold selection",
        "",
        f"Validation-selected isotonic thresholds across the three folds: {', '.join(f'{v:.2f}' for v in threshold_values)}.",
        "",
        "The calibrated model trades recall for higher precision and better probability calibration than raw XGBoost at a fixed 0.50 threshold. The zero classifier's accuracy is inflated by class imbalance; its balanced accuracy is 0.50 and it detects no positive CVO cases.",
        "",
        "### Model-selection interpretation",
        "",
        "- Keep isotonic-calibrated XGBoost with validation-selected threshold as the primary learned-CVO candidate when a more balanced precision/recall trade-off and usable probabilities are desired.",
        "- Raw XGBoost may be preferable when recall is weighted more heavily, but its probability calibration is weaker.",
        "- The validation threshold was selected by the evaluation procedure; it is not yet a cost-optimal threshold for the traffic-control system. A future decision analysis should incorporate the relative cost of missed valuable observations versus unnecessary acquisitions.",
        "",
        "## Figures",
        "",
        "- `plots/traffic_prediction_mae_by_scenario.png`",
        "- `plots/traffic_prediction_rmse_by_scenario.png`",
        "- `plots/traffic_prediction_nonzero_mae_by_scenario.png`",
        "- `plots/traffic_prediction_f1_by_scenario.png`",
        "- `plots/learned_cvo_f1_by_seed.png`",
        "- `plots/learned_cvo_balanced_accuracy_by_seed.png`",
        "- `plots/learned_cvo_brier_score_by_seed.png`",
        "",
        "## Limitations and claims to avoid",
        "",
        "1. Three seeds provide a useful robustness check, but the standard deviations are not a substitute for broad uncertainty estimates from many independent simulation seeds.",
        "2. Traffic prediction is evaluated offline on previously generated SUMO data; it has not yet been shown to improve closed-loop traffic performance when integrated into the controller.",
        "3. Learned CVO estimates the same-timestep action-divergence label defined by the dataset construction. This does not establish future real-world observation value or end-to-end system improvement.",
        "4. Do not claim universal improvement over the zero baseline on overall traffic-prediction MAE, or universal traffic-control improvement from the ML extension.",
        "",
        "## Reproducibility",
        "",
        "The report is generated from the four seed-robustness CSVs already present in this directory. It does not retrain models, overwrite the saved model artifacts, or rerun SUMO.",
        "",
    ])

    report_text = "\n".join(report_lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    summary = {
        "evaluation": "three-fold seed-rotated ML robustness evaluation",
        "temporal_traffic_prediction": {
            "folds": 3,
            "scenario_direction_cells": 12,
            "macro_average_across_cells": macros,
            "temporal_rf_better_cells": counts,
            "mean_relative_reduction_vs_persistence_percent": {
                "MAE": mae_red_persistence,
                "RMSE": rmse_red_persistence,
                "NonzeroMAE": nonzero_red_persistence,
            },
        },
        "learned_cvo": {
            "folds": 3,
            "summary_metrics": {
                model: {
                    column: (None if pd.isna(cvo_index.loc[model, column]) else float(cvo_index.loc[model, column]))
                    for column in cvo.columns if column.endswith("_mean") or column.endswith("_std")
                }
                for model in [zero_name, raw_name, calibrated_name]
            },
            "isotonic_validation_thresholds_by_fold": threshold_values,
        },
        "outputs": {
            "report": str(REPORT_FILE.relative_to(PROJECT_ROOT)),
            "plots_directory": str(PLOTS_DIR.relative_to(PROJECT_ROOT)),
        },
        "no_training_or_sumo_reruns": True,
    }
    JSON_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("ML robustness report generated successfully.")
    print("Report:", REPORT_FILE)
    print("Summary JSON:", JSON_FILE)
    print("Plots:", PLOTS_DIR)
    print("\nTemporal RF vs persistence:")
    print(f"  MAE reduction: {mae_red_persistence:.2f}%")
    print(f"  RMSE reduction: {rmse_red_persistence:.2f}%")
    print(f"  Nonzero-target MAE reduction: {nonzero_red_persistence:.2f}%")
    print("CVO isotonic validation thresholds:", threshold_values)


if __name__ == "__main__":
    main()
