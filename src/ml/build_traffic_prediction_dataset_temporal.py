import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"

OUTPUT_FILE = OUTPUT_DIR / "traffic_prediction_dataset_temporal_v2.csv"

MEASUREMENT_START = 60
MEASUREMENT_END = 600
HORIZON = 10

INTERSECTIONS = ["J1", "J2", "J3", "J4", "J5"]

QUEUE_DIRECTIONS = ["north", "south", "east", "west"]
WAITING_DIRECTIONS = QUEUE_DIRECTIONS

BASE_FEATURES = [
    f"observed_{d}_queue" for d in QUEUE_DIRECTIONS
] + [
    f"observed_{d}_waiting" for d in WAITING_DIRECTIONS
] + [
    "observed_signal_phase",
    "observed_vehicle_count",
]

DELTA_FEATURES = [
    f"queue_change_{d}_per_second" for d in QUEUE_DIRECTIONS
] + [
    f"waiting_change_{d}_per_second" for d in WAITING_DIRECTIONS
]

EXTRA_FEATURES = [
    "observation_age",
    "previous_observation_interval",
    "has_previous_observation",
]

TARGETS = [
    f"target_{d}_queue" for d in QUEUE_DIRECTIONS
] + ["target_network_vehicle_count"]

FEATURES = BASE_FEATURES + DELTA_FEATURES + EXTRA_FEATURES

RUN_PATTERN = re.compile(
    r"^network__(low|medium|high)__"
    r"(always|periodic|state_change|aoi|cvo)__seed0([1-3])$"
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key_time(value):
    return round(float(value), 6)


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT_FILE}"
        )

    run_dirs = []

    for directory in sorted(RAW_DIR.glob("network__*__seed0*")):
        match = RUN_PATTERN.fullmatch(directory.name)
        if match:
            run_dirs.append((directory, match))

    if len(run_dirs) != 45:
        raise ValueError(
            f"Expected 45 network runs, found {len(run_dirs)}"
        )

    output_rows = []
    skipped = 0

    for run_dir, match in run_dirs:
        scenario, strategy, seed = match.groups()

        traffic_rows = read_csv(run_dir / "traffic.csv")
        acquisition_rows = read_csv(run_dir / "acquisition.csv")

        states = {i: {} for i in INTERSECTIONS}
        decisions = {}

        for row in traffic_rows:
            intersection = row["intersection_id"]
            timestamp = key_time(row["time"])

            if intersection not in states:
                raise ValueError(
                    f"{run_dir.name}: unknown intersection {intersection}"
                )

            if timestamp in states[intersection]:
                raise ValueError(
                    f"{run_dir.name}: duplicate traffic timestamp"
                )

            states[intersection][timestamp] = row

        for row in acquisition_rows:
            key = (
                row["intersection_id"],
                key_time(row["time"]),
            )

            if key in decisions:
                raise ValueError(
                    f"{run_dir.name}: duplicate acquisition record"
                )

            decisions[key] = row["decision"]

        traffic_keys = {
            (intersection, timestamp)
            for intersection, records in states.items()
            for timestamp in records
        }

        if traffic_keys != set(decisions):
            raise ValueError(
                f"{run_dir.name}: traffic/acquisition records mismatch"
            )

        for intersection in INTERSECTIONS:

            previous_observation = None
            previous_observation_time = None

            last_observation = None
            last_observation_time = None

            for timestamp in sorted(states[intersection]):

                true_state = states[intersection][timestamp]

                decision = decisions[(intersection, timestamp)]

                if decision == "ACQUIRE":

                    previous_observation = last_observation
                    previous_observation_time = last_observation_time

                    last_observation = true_state
                    last_observation_time = timestamp

                elif decision != "SKIP":
                    raise ValueError(
                        f"Unexpected acquisition decision: {decision}"
                    )

                if not (
                    MEASUREMENT_START <= timestamp
                    and timestamp + HORIZON < MEASUREMENT_END
                ):
                    continue

                target_time = key_time(timestamp + HORIZON)

                future_state = states[intersection].get(target_time)

                if last_observation is None or future_state is None:
                    skipped += 1
                    continue

                has_previous = previous_observation is not None

                interval = (
                    last_observation_time - previous_observation_time
                    if has_previous else 0.0
                )

                safe_interval = max(interval, 1.0)

                output = {
                    "run_id": run_dir.name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "seed": int(seed),
                    "intersection_id": intersection,
                    "time": timestamp,
                    "target_time": target_time,
                    "last_observation_time": last_observation_time,
                    "observation_age": timestamp - last_observation_time,
                    "previous_observation_interval": interval,
                    "has_previous_observation": int(has_previous),
                }

                for column in BASE_FEATURES:
                    raw_column = column.replace("observed_", "")
                    output[column] = last_observation[raw_column]

                for direction in QUEUE_DIRECTIONS:

                    current_queue = float(
                        last_observation[f"{direction}_queue"]
                    )

                    previous_queue = (
                        float(previous_observation[f"{direction}_queue"])
                        if has_previous else current_queue
                    )

                    output[
                        f"queue_change_{direction}_per_second"
                    ] = (
                        current_queue - previous_queue
                    ) / safe_interval

                    current_waiting = float(
                        last_observation[f"{direction}_waiting"]
                    )

                    previous_waiting = (
                        float(previous_observation[f"{direction}_waiting"])
                        if has_previous else current_waiting
                    )

                    output[
                        f"waiting_change_{direction}_per_second"
                    ] = (
                        current_waiting - previous_waiting
                    ) / safe_interval

                    output[f"target_{direction}_queue"] = (
                        future_state[f"{direction}_queue"]
                    )

                output["target_network_vehicle_count"] = (
                    future_state["vehicle_count"]
                )

                output_rows.append(output)

    if not output_rows:
        raise ValueError("No temporal training samples generated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id", "scenario", "strategy", "seed",
        "intersection_id", "time", "target_time",
        "last_observation_time", "observation_age",
        "previous_observation_interval",
        "has_previous_observation",
    ] + FEATURES[:-3] + TARGETS

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 65)
    print("TEMPORAL TRAFFIC PREDICTION DATASET CREATED")
    print("=" * 65)
    print("Runs processed:", len(run_dirs))
    print("Prediction horizon:", HORIZON, "seconds")
    print("Samples:", len(output_rows))
    print("Input features:", len(FEATURES))
    print("Prediction targets:", len(TARGETS))
    print("Skipped samples:", skipped)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
