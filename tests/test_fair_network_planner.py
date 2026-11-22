import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.acquisition.network_planner import NetworkObservationPlanner


def main():
    ids = ["J1", "J2", "J3", "J4"]
    cvo = {i: 0.5 for i in ids}
    costs = {i: 0.4 for i in ids}
    margins = {"J1": 100, "J2": 90, "J3": 80, "J4": 70}
    ages = {"J1": 1, "J2": 4, "J3": 3, "J4": 0}

    # Default mode preserves original CVO-then-margin ordering.
    original_planner = NetworkObservationPlanner(observation_budget=2)
    original_selected = original_planner.select_intersections(
        cvo, costs, margins, ages
    )
    assert original_selected == ["J1", "J2"], original_selected

    # Experimental mode uses age before margin when CVO values tie.
    fair_planner = NetworkObservationPlanner(
        observation_budget=2,
        age_priority_on_ties=True,
    )
    fair_selected = fair_planner.select_intersections(
        cvo, costs, margins, ages
    )
    assert fair_selected == ["J2", "J3"], fair_selected

    # Beliefs at/above the AoI threshold are prioritized first.
    forced_selected = fair_planner.select_intersections(
        cvo_values=cvo,
        cost_values=costs,
        decision_margins=margins,
        belief_ages={"J1": 0, "J2": 11, "J3": 12, "J4": 0},
    )
    assert forced_selected == ["J3", "J2"], forced_selected

    # The normalized CVO score must exceed cost to be eligible.
    low_cvo = {i: 0.25 for i in ids}
    none_selected = fair_planner.select_intersections(
        cvo_values=low_cvo,
        cost_values=costs,
        decision_margins=margins,
        belief_ages={i: 0 for i in ids},
    )
    assert none_selected == [], none_selected

    # Budget remains a hard upper bound.
    assert len(fair_selected) <= 2
    print("Fair network planner tests passed: 5/5")


if __name__ == "__main__":
    main()
