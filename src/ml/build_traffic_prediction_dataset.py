import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"
OUTPUT_FILE = OUTPUT_DIR / "traffic_prediction_dataset.csv"

MEASUREMENT_START = 60
MEASUREMENT_END = 600
PREDICTION_HORIZON = 10

INTERSECTIONS = {"J1", "J2", "J3", "J4", "J5"}
SCENARIOS = {"low", "medium", "high"}
STRATEGIES = {"always", "periodic", "state_change", "aoi", "cvo"}

RUN_PATTERN = re.compile(
    r"^network__(low|medium|high)__"
    r"(always|periodic|state_change|aoi|cvo)__seed0([1-3])$"
)

QUEUE_COLUMNS = [
    "north_queue", "south_queue", "east_queue", "west_queue"
]

WAITING_COLUMNS = [
    "north_waiting", "south_waiting", "east_waiting", "west_waiting"
]

OBSERVED_COLUMNS = (
    QUEUE_COLUMNS
    + WAITING_COLUMNS
    + ["signal_phase", "vehicle_count"]
)

TARGET_COLUMNS = [
    f"target_{direction}_queue"
    for direction in ["north", "south", "east", "west"]
] + ["target_network_vehicle_count"]

OUTPUT_COLUMNS = [
    "run_id", "scenario", "strategy", "seed",
    "time", "target_time", "intersection_id",
    "last_observation_time", "observation_age",
] + [
    f"observed_{column}" for column in OBSERVED_COLUMNS
] + TARGET_COLUMNS


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def time_key(value):
    return round(float(value), 6)


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {OUTPUT_FILE}"
        )

    run_dirs = []

    for directory in sorted(RAW_DIR.glob("network__*__seed0*")):
        match = RUN_PATTERN.fullmatch(directory.name)
        if match:
            run_dirs.append((directory, match))

    if len(run_dirs) != 45:
        raise ValueError(
            f"Expected 45 network run folders, found {len(run_dirs)}."
        )

    dataset = []
    skipped_without_observation = 0
    skipped_without_target = 0

    for run_dir, match in run_dirs:
        scenario, strategy, seed = match.groups()

        traffic_rows = read_csv(run_dir / "traffic.csv")
        acquisition_rows = read_csv(run_dir / "acquisition.csv")

        traffic_by_intersection = {i: {} for i in INTERSECTIONS}
        acquisition_by_key = {}

        for row in traffic_rows:
            intersection = row["intersection_id"]
            if intersection not in INTERSECTIONS:
                raise ValueError(
                    f"{run_dir.name}: unknown intersection {intersection}"
                )

            timestamp = time_key(row["time"])
            if timestamp in traffic_by_intersection[intersection]:
                raise ValueError(
                    f"{run_dir.name}: duplicate traffic row "
                    f"for {intersection} at {timestamp}"
                )

            traffic_by_intersection[intersection][timestamp] = row

        for row in acquisition_rows:
            intersection = row["intersection_id"]
            timestamp = time_key(row["time"])
            key = (intersection, timestamp)

            if key in acquisition_by_key:
                raise ValueError(
                    f"{run_dir.name}: duplicate acquisition record {key}"
                )

            acquisition_by_key[key] = row["decision"]

        traffic_keys = {
            (intersection, timestamp)
            for intersection, states in traffic_by_intersection.items()
            for timestamp in states
        }

        if traffic_keys != set(acquisition_by_key):
            raise ValueError(
                f"{run_dir.name}: traffic and acquisition timestamps "
                "do not match."
            )

        for intersection in sorted(INTERSECTIONS):
            states = traffic_by_intersection[intersection]
            last_observation = None
            last_observation_time = None

            for timestamp in sorted(states):
                true_state = states[timestamp]
                decision = acquisition_by_key[(intersection, timestamp)]

                if decision == "ACQUIRE":
                    last_observation = true_state
                    last_observation_time = timestamp
                elif decision != "SKIP":
                    raise ValueError(
                        f"{run_dir.name}: unexpected decision {decision}"
                    )

                target_time = time_key(timestamp + PREDICTION_HORIZON)

                if not (
                    MEASUREMENT_START <= timestamp
                    and target_time < MEASUREMENT_END
                ):
                    continue

                if last_observation is None:
                    skipped_without_observation += 1
                    continue

                future_state = states.get(target_time)
                if future_state is None:
                    skipped_without_target += 1
                    continue

                output_row = {
                    "run_id": run_dir.name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "seed": seed,
                    "time": timestamp,
                    "target_time": target_time,
                    "intersection_id": intersection,
                    "last_observation_time": last_observation_time,
                    "observation_age": round(
                        timestamp - last_observation_time, 6
                    ),
                }

                for column in OBSERVED_COLUMNS:
                    output_row[f"observed_{column}"] = (
                        last_observation[column]
                    )

                for direction in ["north", "south", "east", "west"]:
                    output_row[f"target_{direction}_queue"] = (
                        future_state[f"{direction}_queue"]
                    )

                output_row["target_network_vehicle_count"] = (
                    future_state["vehicle_count"]
                )

                dataset.append(output_row)

    if not dataset:
        raise ValueError("No prediction samples were generated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(dataset)

    print("=" * 65)
    print("TRAFFIC PREDICTION DATASET CREATED")
    print("=" * 65)
    print(f"Network runs processed: {len(run_dirs)}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} seconds")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Features: {len(OBSERVED_COLUMNS) + 1}")
    print(f"Prediction targets: {len(TARGET_COLUMNS)}")
    print(f"Skipped (no acquired observation): {skipped_without_observation}")
    print(f"Skipped (missing future target): {skipped_without_target}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
