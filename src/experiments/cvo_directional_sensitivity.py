import csv
import json
from collections import defaultdict
from pathlib import Path

from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief
from src.control.network_controller import NetworkSignalController


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"

SCENARIOS = ["low", "medium", "high"]
SEEDS = [1, 2, 3]
DIRECTIONS = [
    ("NS +5", "north_queue"),
    ("EW +5", "east_queue"),
]

MARGIN_BINS = ["0-<2", "2-<5", "5-<10", "10-<20", "20+"]


def get_margin_bin(margin):
    if margin < 2:
        return "0-<2"
    if margin < 5:
        return "2-<5"
    if margin < 10:
        return "5-<10"
    if margin < 20:
        return "10-<20"
    return "20+"


def load_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    counts = defaultdict(lambda: {"observations": 0, "changes": 0})
    action_mismatches = 0
    total_rows = 0

    for scenario in SCENARIOS:
        for seed in SEEDS:
            run_id = f"network__{scenario}__cvo__seed0{seed}"
            cvo_file = RAW_DIR / run_id / "cvo.csv"

            if not cvo_file.exists():
                raise FileNotFoundError(cvo_file)

            rows = load_rows(cvo_file)

            for row in rows:
                time = float(row["time"])

                if not (60 <= time < 600):
                    continue

                total_rows += 1
                intersection_id = row["intersection_id"]
                controller = NetworkSignalController(intersection_id)

                state_data = json.loads(row["current_belief"])
                last_observation_time = state_data.pop(
                    "last_observation_time"
                )

                current_state = TrafficState(**state_data)
                current_belief = Belief(
                    state=current_state,
                    last_observation_time=last_observation_time,
                )

                current_action = controller.choose_action(current_belief)

                if current_action != row["current_action"]:
                    action_mismatches += 1

                margin = float(row["decision_margin"])
                margin_bin = get_margin_bin(margin)

                for direction, queue_field in DIRECTIONS:
                    hypothetical_data = dict(state_data)
                    hypothetical_data[queue_field] = (
                        float(hypothetical_data[queue_field]) + 5
                    )

                    hypothetical_state = TrafficState(
                        **hypothetical_data
                    )
                    hypothetical_belief = Belief(
                        state=hypothetical_state,
                        last_observation_time=last_observation_time,
                    )

                    hypothetical_action = controller.choose_action(
                        hypothetical_belief
                    )

                    changed = int(
                        hypothetical_action != current_action
                    )

                    key = (scenario, direction, margin_bin)
                    counts[key]["observations"] += 1
                    counts[key]["changes"] += changed

    summary = []

    for scenario in SCENARIOS:
        for direction, _ in DIRECTIONS:
            for margin_bin in MARGIN_BINS:
                key = (scenario, direction, margin_bin)
                group = counts[key]

                observations = group["observations"]
                changes = group["changes"]

                change_pct = (
                    round(100.0 * changes / observations, 2)
                    if observations
                    else None
                )

                summary.append({
                    "Scenario": scenario,
                    "Perturbation": direction,
                    "MarginBin": margin_bin,
                    "Observations": observations,
                    "ActionChanges": changes,
                    "ActionChangePct": change_pct,
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "cvo_directional_sensitivity.csv"

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
    print(f"Measurement rows checked: {total_rows}")
    print(f"Current-action mismatches: {action_mismatches}")
    print()
    print(
        f"{'Scenario':<9} {'Perturbation':<14} "
        f"{'Margin':<9} {'Observations':>12} "
        f"{'Changes':>9} {'Change %':>10}"
    )

    for row in summary:
        pct = (
            f"{row['ActionChangePct']:.2f}"
            if row["ActionChangePct"] is not None
            else "N/A"
        )

        print(
            f"{row['Scenario']:<9} "
            f"{row['Perturbation']:<14} "
            f"{row['MarginBin']:<9} "
            f"{row['Observations']:>12} "
            f"{row['ActionChanges']:>9} "
            f"{pct:>10}"
        )


if __name__ == "__main__":
    main()