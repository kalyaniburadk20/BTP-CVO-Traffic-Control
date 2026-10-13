import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "processed"
    / "network_experiment_statistics.csv"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "results"
    / "processed"
    / "plots"
)

SCENARIOS = ["low", "medium", "high"]

STRATEGIES = [
    "always",
    "periodic",
    "state_change",
    "aoi",
    "cvo",
]

STRATEGY_LABELS = {
    "always": "Always",
    "periodic": "Periodic",
    "state_change": "State Change",
    "aoi": "AoI",
    "cvo": "CVO",
}

METRICS = [
    {
        "mean_column": "AcquisitionPctMean",
        "sd_column": "AcquisitionPctSD",
        "title": "Observation Acquisition Rate",
        "ylabel": "Acquisition rate (%)",
        "filename": "acquisition_rate.png",
    },
    {
        "mean_column": "AverageQueueMean",
        "sd_column": "AverageQueueSD",
        "title": "Mean Stopped-Vehicle Count per Intersection",
        "ylabel": "Mean stopped vehicles across four approaches",
        "filename": "average_queue.png",
    },
    {
        "mean_column": "AverageWaitingMean",
        "sd_column": "AverageWaitingSD",
        "title": "Mean Approach-Level Waiting-Time Burden",
        "ylabel": "Mean summed current waiting time (seconds)",
        "filename": "average_waiting.png",
    },
    {
        "mean_column": "ThroughputMean",
        "sd_column": "ThroughputSD",
        "title": "Network Throughput",
        "ylabel": "Arrived vehicles during measurement",
        "filename": "throughput.png",
    },
]


def load_summary():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Summary file not found: {SUMMARY_FILE}"
        )

    with SUMMARY_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    expected_rows = len(SCENARIOS) * len(STRATEGIES)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found {len(rows)}."
        )

    return rows


def create_metric_plot(rows, metric):
    fig, ax = plt.subplots(figsize=(11, 6))

    group_positions = list(range(len(SCENARIOS)))
    bar_width = 0.15

    for strategy_index, strategy in enumerate(STRATEGIES):
        values = []
        errors = []

        for scenario in SCENARIOS:
            matching_rows = [
                row
                for row in rows
                if row["Scenario"] == scenario
                and row["Strategy"] == strategy
            ]

            if len(matching_rows) != 1:
                raise ValueError(
                    f"Expected one row for {scenario}/{strategy}."
                )

            values.append(
                float(matching_rows[0][metric["mean_column"]])
            )
            errors.append(
                float(matching_rows[0][metric["sd_column"]])
            )

        offsets = [
            position
            + (strategy_index - (len(STRATEGIES) - 1) / 2)
            * bar_width
            for position in group_positions
        ]

        ax.bar(
            offsets,
            values,
            bar_width,
            label=STRATEGY_LABELS[strategy],
            yerr=errors,
            capsize=4,
        )

    ax.set_title(metric["title"])
    ax.set_xlabel("Traffic scenario")
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(group_positions)
    ax.set_xticklabels(
        [scenario.capitalize() for scenario in SCENARIOS]
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False)

    fig.tight_layout()

    output_path = PLOT_DIR / metric["filename"]

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_summary()

    for metric in METRICS:
        create_metric_plot(rows, metric)

    print()
    print("All four plots created successfully.")


if __name__ == "__main__":
    main()