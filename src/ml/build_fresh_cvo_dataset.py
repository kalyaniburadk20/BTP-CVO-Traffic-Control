import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.control.network_controller import NetworkSignalController


RAW_DIR = Path("results/raw")
OUTPUT_DIR = Path("results/processed/ml")
OUTPUT_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_dataset.csv"

STATE_FIELDS = [
    "north_queue",
    "south_queue",
    "east_queue",
    "west_queue",
    "north_waiting",
    "south_waiting",
    "east_waiting",
    "west_waiting",
    "current_phase",
    "vehicle_count",
]

TRAFFIC_FIELDS = [
    "north_queue",
    "south_queue",
    "east_queue",
    "west_queue",
    "north_waiting",
    "south_waiting",
    "east_waiting",
    "west_waiting",
    "vehicle_count",
]

FEATURES = [
    *[f"current_{field}" for field in STATE_FIELDS],
    "belief_age",
    "decision_margin",
    "scenario",
    "intersection_id",
]


def choose_action(controller, state_values):
    belief = SimpleNamespace(
        state=SimpleNamespace(**state_values)
    )
    return controller.choose_action(belief)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT_PATH}"
        )

    run_pattern = re.compile(
        r"network__(low|medium|high)__cvo__seed0([1-3])$"
    )

    run_directories = []

    for directory in sorted(RAW_DIR.iterdir()):
        if directory.is_dir() and run_pattern.fullmatch(directory.name):
            run_directories.append(directory)

    if len(run_directories) != 9:
        raise ValueError(
            f"Expected 9 CVO run folders, found {len(run_directories)}."
        )

    controllers = {}
    records = []
    total_matched = 0

    for run_directory in run_directories:
        match = run_pattern.fullmatch(run_directory.name)
        scenario = match.group(1)
        seed = int(match.group(2))

        cvo_path = run_directory / "cvo.csv"
        traffic_path = run_directory / "traffic.csv"

        if not cvo_path.exists() or not traffic_path.exists():
            raise FileNotFoundError(
                f"Required log missing in {run_directory}"
            )

        cvo_df = pd.read_csv(cvo_path)
        traffic_df = pd.read_csv(traffic_path)

        if cvo_df.empty:
            raise ValueError(f"Empty CVO log: {cvo_path}")

        required_cvo = {
            "time",
            "intersection_id",
            "current_belief",
            "decision_margin",
            "cvo",
        }
        required_traffic = {
            "time",
            "intersection_id",
            *TRAFFIC_FIELDS,
        }

        if not required_cvo.issubset(cvo_df.columns):
            raise ValueError(
                f"Missing CVO columns in {cvo_path}"
            )

        if not required_traffic.issubset(traffic_df.columns):
            raise ValueError(
                f"Missing traffic columns in {traffic_path}"
            )

        keys = ["time", "intersection_id"]

        if cvo_df.duplicated(keys).any():
            raise ValueError(
                f"Duplicate CVO timestamps in {run_directory.name}"
            )

        if traffic_df.duplicated(keys).any():
            raise ValueError(
                f"Duplicate traffic timestamps in {run_directory.name}"
            )

        merged = cvo_df.merge(
            traffic_df,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(cvo_df):
            raise ValueError(
                f"Unmatched CVO/traffic records in {run_directory.name}: "
                f"{len(cvo_df) - len(merged)}"
            )

        total_matched += len(merged)

        for row in merged.to_dict("records"):
            intersection_id = str(row["intersection_id"])

            if intersection_id not in controllers:
                controllers[intersection_id] = NetworkSignalController(
                    intersection_id
                )

            controller = controllers[intersection_id]

            current_state = json.loads(row["current_belief"])

            if not isinstance(current_state, dict):
                raise ValueError("Unexpected current_belief format.")

            missing_state = set(STATE_FIELDS) - set(current_state)
            if missing_state:
                raise ValueError(
                    f"Missing belief fields: {sorted(missing_state)}"
                )

            last_observation_time = current_state.get(
                "last_observation_time"
            )

            if last_observation_time is None:
                raise ValueError("Missing last_observation_time.")

            timestamp = float(row["time"])
            belief_age = timestamp - float(last_observation_time)

            if belief_age < 0:
                raise ValueError("Negative belief age found.")

            # Current action: use the original pre-acquisition belief.
            current_action = choose_action(
                controller,
                {
                    field: current_state[field]
                    for field in STATE_FIELDS
                },
            )

            # Fresh queue state: use actual SUMO measurements.
            # Keep the old belief phase fixed to isolate queue-driven
            # action divergence and avoid post-action phase leakage.
            fresh_state = {
                field: row[field]
                for field in TRAFFIC_FIELDS
            }
            fresh_state["current_phase"] = current_state["current_phase"]

            fresh_action = choose_action(
                controller,
                fresh_state,
            )

            label = int(current_action != fresh_action)

            record = {
                "run_id": run_directory.name,
                "scenario": scenario,
                "seed": seed,
                "time": timestamp,
                "intersection_id": intersection_id,
                **{
                    f"current_{field}": current_state[field]
                    for field in STATE_FIELDS
                },
                "belief_age": belief_age,
                "decision_margin": float(row["decision_margin"]),
                "current_action": current_action,
                "fresh_queue_action": fresh_action,
                "fresh_cvo": label,
            }

            records.append(record)

    dataset = pd.DataFrame(records)

    if dataset.empty:
        raise ValueError("No fresh-queue samples were generated.")

    if dataset.duplicated(
        subset=["run_id", "intersection_id", "time"]
    ).any():
        raise ValueError("Duplicate generated samples found.")

    if dataset[FEATURES + ["fresh_cvo"]].isna().any().any():
        raise ValueError("Missing values found in generated dataset.")

    if not set(dataset["fresh_cvo"].unique()).issubset({0, 1}):
        raise ValueError("Invalid fresh_cvo labels.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("FRESH-QUEUE CVO DATASET CREATED")
    print("=" * 65)
    print(f"CVO runs processed: {len(run_directories)}")
    print(f"Matched samples: {total_matched}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Input features: {len(FEATURES)}")
    print(f"Fresh CVO = 0: {(dataset['fresh_cvo'] == 0).sum()}")
    print(f"Fresh CVO = 1: {(dataset['fresh_cvo'] == 1).sum()}")
    print(
        "Current/fresh actions equal:",
        (dataset["current_action"] == dataset["fresh_queue_action"]).sum(),
    )
    print("Missing values:", dataset.isna().sum().sum())
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
