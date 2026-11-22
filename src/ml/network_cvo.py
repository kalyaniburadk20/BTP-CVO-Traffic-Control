"""Adapter between live network beliefs and the calibrated fresh-CVO model.

The model estimates the probability that fresh queue measurements would
change the signal controller's action. The probability is a CVO score for
the acquisition planner; it is not an acquisition command by itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.cvo.cvo import calculate_decision_margin
from src.ml.inference import (
    load_fresh_cvo_model,
    predict_fresh_cvo,
)


VALID_SCENARIOS = {"low", "medium", "high"}


def build_fresh_cvo_features(
    belief: Any,
    current_time: float,
    scenario: str,
    intersection_id: str,
) -> dict[str, Any]:
    """Build one model input row from the currently available belief."""
    if belief is None or belief.state is None:
        raise ValueError("A belief containing a traffic state is required.")

    if belief.last_observation_time is None:
        raise ValueError("Belief must have a valid observation timestamp.")

    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"scenario must be one of {sorted(VALID_SCENARIOS)}."
        )

    if not intersection_id:
        raise ValueError("intersection_id must be a non-empty string.")

    age = float(current_time) - float(belief.last_observation_time)
    if age < 0:
        raise ValueError("current_time cannot precede the belief timestamp.")

    state = belief.state
    return {
        "current_north_queue": state.north_queue,
        "current_south_queue": state.south_queue,
        "current_east_queue": state.east_queue,
        "current_west_queue": state.west_queue,
        "current_north_waiting": state.north_waiting,
        "current_south_waiting": state.south_waiting,
        "current_east_waiting": state.east_waiting,
        "current_west_waiting": state.west_waiting,
        "current_current_phase": state.current_phase,
        "current_vehicle_count": state.vehicle_count,
        "belief_age": age,
        "decision_margin": calculate_decision_margin(belief),
        "scenario": scenario,
        "intersection_id": intersection_id,
    }


def predict_network_fresh_cvo(
    beliefs: Mapping[str, Any],
    current_time: float,
    scenario: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return calibrated fresh-CVO probabilities for all supplied beliefs.

    The artifact can be loaded once and reused across simulation timesteps.
    Results are probabilities in [0, 1], keyed by intersection ID.
    """
    if not beliefs:
        raise ValueError("At least one intersection belief is required.")

    rows = [
        build_fresh_cvo_features(
            belief=belief,
            current_time=current_time,
            scenario=scenario,
            intersection_id=intersection_id,
        )
        for intersection_id, belief in beliefs.items()
    ]

    model_artifact = artifact or load_fresh_cvo_model()
    features = pd.DataFrame(rows)
    predictions = predict_fresh_cvo(features, model_artifact)

    probabilities = predictions["fresh_cvo_probability"].to_numpy(
        dtype=float
    )
    if len(probabilities) != len(beliefs):
        raise ValueError("Fresh-CVO model returned an unexpected row count.")

    return {
        intersection_id: float(probability)
        for intersection_id, probability in zip(beliefs.keys(), probabilities)
    }
