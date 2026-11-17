import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"

OUTPUT_FILE = OUTPUT_DIR / "learned_cvo_dataset.csv"

RUN_PATTERN = re.compile(
    r"^network__(low|medium|high)__cvo__seed0([1-3])$"
)

DIRECTIONS = ["north", "south", "east", "west"]

STATE_FIELDS = [
    f"{direction}_queue" for direction in DIRECTIONS
] + [
    f"{direction}_waiting" for direction in DIRECTIONS
] + [
    "current_phase",
    "vehicle_count",
]

FEATURE_COLUMNS = (
    [f"current_{field}" for field in STATE_FIELDS]
    + [f"hypothetical_{field}" for field in STATE_FIELDS]
    + [
        "belief_age",
        "decision_margin",
        "scenario",
        "intersection_id",
    ]
)

METADATA_COLUMNS = [
    "run_id",
    "seed",
    "time",
]

OUTPUT_COLUMNS = (
    METADATA_COLUMNS
    + FEATURE_COLUMNS
    + ["cvo"]
)


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


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

    if len(run_dirs) != 9:
        raise ValueError(
            f"Expected 9 CVO run folders, found {len(run_dirs)}."
        )

    dataset = []
    labels = {0: 0, 1: 0}
    seen = set()

    for run_dir, match in run_dirs:

        scenario, seed = match.groups()

        rows = read_csv(run_dir / "cvo.csv")

        if not rows:
            raise ValueError(f"Empty CVO log: {run_dir.name}")

        for row in rows:

            current = json.loads(row["current_belief"])
            hypothetical = json.loads(row["hypothetical_belief"])

            timestamp = float(row["time"])
            intersection = row["intersection_id"]

            key = (
                run_dir.name,
                intersection,
                timestamp,
            )

            if key in seen:
                raise ValueError(
                    f"Duplicate CVO record: {key}"
                )

            seen.add(key)

            label = int(float(row["cvo"]))

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid CVO label in {run_dir.name}"
                )

            expected_label = int(
                row["current_action"]
                != row["hypothetical_action"]
            )

            if label != expected_label:
                raise ValueError(
                    f"Action-label mismatch: {key}"
                )

            if "last_observation_time" not in current:
                raise ValueError(
                    f"Missing observation timestamp: {key}"
                )

            output = {
                "run_id": run_dir.name,
                "seed": int(seed),
                "time": timestamp,
                "scenario": scenario,
                "intersection_id": intersection,
                "belief_age": (
                    timestamp
                    - float(current["last_observation_time"])
                ),
                "decision_margin": float(row["decision_margin"]),
                "cvo": label,
            }

            for field in STATE_FIELDS:
                output[f"current_{field}"] = current[field]
                output[f"hypothetical_{field}"] = hypothetical[field]

            dataset.append(output)
            labels[label] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(dataset)

    print("=" * 65)
    print("LEARNED CVO DATASET CREATED")
    print("=" * 65)
    print("CVO runs processed:", len(run_dirs))
    print("Total samples:", len(dataset))
    print("Input features:", len(FEATURE_COLUMNS))
    print("CVO = 0:", labels[0])
    print("CVO = 1:", labels[1])
    print("Action-label inconsistencies: 0")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
