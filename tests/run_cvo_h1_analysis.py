import os
import sys
import csv


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief

from src.control.controller import SignalController

from src.cvo.cvo import calculate_cvo


# --------------------------------------------------
# OUTPUT FILE
# --------------------------------------------------

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw",
    "cvo_h1_boundary_analysis.csv"
)


# --------------------------------------------------
# CREATE BELIEF
# --------------------------------------------------

def create_belief(ns_pressure, ew_pressure):

    state = TrafficState(
        north_queue=ns_pressure,
        south_queue=0,
        east_queue=ew_pressure,
        west_queue=0,
        north_waiting=0.0,
        south_waiting=0.0,
        east_waiting=0.0,
        west_waiting=0.0,
        current_phase=0,
        vehicle_count=0
    )

    return Belief(
        state=state,
        last_observation_time=0.0
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 100)
    print("PHASE 6 — H1 DECISION-BOUNDARY ANALYSIS")
    print("=" * 100)

    controller = SignalController()

    # --------------------------------------------------
    # Current decision margins to investigate
    # --------------------------------------------------

    margins = [1, 2, 3, 5, 10, 20, 30, 50]

    # --------------------------------------------------
    # Hypothetical changes in EW pressure
    # --------------------------------------------------

    perturbations = [0, 1, 2, 3, 5, 10]

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # --------------------------------------------------
    # Open CSV
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "current_margin",
                "hypothetical_ew_change",
                "current_action",
                "hypothetical_action",
                "action_changed",
                "cvo"
            ]
        )

        writer.writeheader()

        # --------------------------------------------------
        # Run experiment
        # --------------------------------------------------

        for margin in margins:

            # Current state:
            #
            # NS = 20 + margin
            # EW = 20
            #
            # Therefore:
            # margin = |NS - EW|
            #
            current_belief = create_belief(
                ns_pressure=20 + margin,
                ew_pressure=20
            )

            current_action = controller.choose_action(
                current_belief
            )

            for perturbation in perturbations:

                # --------------------------------------------------
                # Hypothetical observation
                # --------------------------------------------------

                hypothetical_belief = create_belief(
                    ns_pressure=20 + margin,
                    ew_pressure=20 + perturbation
                )

                hypothetical_action = (
                    controller.choose_action(
                        hypothetical_belief
                    )
                )

                # --------------------------------------------------
                # CVO
                # --------------------------------------------------

                cvo = calculate_cvo(
                    current_belief,
                    hypothetical_belief,
                    controller
                )

                action_changed = (
                    current_action != hypothetical_action
                )

                # --------------------------------------------------
                # Save
                # --------------------------------------------------

                writer.writerow({
                    "current_margin": margin,
                    "hypothetical_ew_change": perturbation,
                    "current_action": current_action,
                    "hypothetical_action": hypothetical_action,
                    "action_changed": action_changed,
                    "cvo": cvo
                })

                # --------------------------------------------------
                # Print
                # --------------------------------------------------

                print(
                    f"Margin={margin:>2} | "
                    f"EW change={perturbation:>2} | "
                    f"Current={current_action:>9} | "
                    f"Hypothetical={hypothetical_action:>9} | "
                    f"Changed={str(action_changed):<5} | "
                    f"CVO={cvo:.1f}"
                )

    print()
    print("-" * 100)
    print(
        f"Saved analysis to:\n{OUTPUT_FILE}"
    )

    print()
    print(
        "✅ H1 DECISION-BOUNDARY ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()