import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"

SCENARIOS = ["low", "medium", "high"]
STRATEGIES = ["always", "periodic", "state_change", "aoi", "cvo"]
SEEDS = [1, 2, 3]

MEASUREMENT_START = 60
MEASUREMENT_END = 600
EXPECTED_ROWS_PER_RUN = 2700


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def mean_and_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def analyze_run(scenario, strategy, seed):
    run_id = f"network__{scenario}__{strategy}__seed0{seed}"
    run_dir = RAW_DIR / run_id

    acquisition_rows = read_csv(run_dir / "acquisition.csv")
    traffic_rows = read_csv(run_dir / "traffic.csv")

    acquisition_rows = [
        row for row in acquisition_rows
        if MEASUREMENT_START <= float(row["time"]) < MEASUREMENT_END
    ]

    traffic_rows = [
        row for row in traffic_rows
        if MEASUREMENT_START <= float(row["time"]) < MEASUREMENT_END
    ]

    if len(acquisition_rows) != EXPECTED_ROWS_PER_RUN:
        raise ValueError(
            f"{run_id}: expected {EXPECTED_ROWS_PER_RUN} "
            f"acquisition rows, found {len(acquisition_rows)}"
        )

    if len(traffic_rows) != EXPECTED_ROWS_PER_RUN:
        raise ValueError(
            f"{run_id}: expected {EXPECTED_ROWS_PER_RUN} "
            f"traffic rows, found {len(traffic_rows)}"
        )

    acquired = sum(
        row["decision"] == "ACQUIRE"
        for row in acquisition_rows
    )

    acquisition_pct = 100.0 * acquired / len(acquisition_rows)

    queue_values = [
        sum(
            float(row[f"{direction}_queue"])
            for direction in ["north", "south", "east", "west"]
        )
        for row in traffic_rows
    ]

    waiting_values = [
        sum(
            float(row[f"{direction}_waiting"])
            for direction in ["north", "south", "east", "west"]
        )
        for row in traffic_rows
    ]

    # arrived_this_step is repeated across intersection rows.
    # Count it only once for each simulation timestep.
    arrivals_by_time = {}

    for row in traffic_rows:
        arrivals_by_time.setdefault(
            float(row["time"]),
            float(row["arrived_this_step"]),
        )

    throughput = sum(arrivals_by_time.values())

    return {
        "AcquisitionPct": acquisition_pct,
        "AverageQueue": statistics.mean(queue_values),
        "AverageWaiting": statistics.mean(waiting_values),
        "Throughput": throughput,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "AcquisitionPct",
        "AverageQueue",
        "AverageWaiting",
        "Throughput",
    ]

    summary = []

    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            seed_results = [
                analyze_run(scenario, strategy, seed)
                for seed in SEEDS
            ]

            row = {
                "Scenario": scenario,
                "Strategy": strategy,
            }

            for metric in metric_names:
                values = [result[metric] for result in seed_results]
                mean, sd = mean_and_sd(values)

                row[f"{metric}Mean"] = round(mean, 4)
                row[f"{metric}SD"] = round(sd, 4)

            summary.append(row)

    output_file = OUTPUT_DIR / "network_experiment_summary.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved: {output_file}")
    print(f"Summary rows: {len(summary)}")
    print("Seeds per scenario-strategy group: 3")


if __name__ == "__main__":
    main()