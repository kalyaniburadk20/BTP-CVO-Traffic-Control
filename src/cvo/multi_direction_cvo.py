"""Directional hypothetical perturbations for experimental CVO analysis.

This module is deliberately separate from HypotheticalObservationGenerator:
the original fixed east-queue (+5) experiment remains unchanged and reproducible.
"""

from src.cvo.action_divergence import action_divergence
from src.perception.belief import Belief


class MultiDirectionHypotheticalGenerator:
    """Build one independent hypothetical belief per incoming direction."""

    DIRECTIONS = ("north", "south", "east", "west")

    def __init__(self, queue_delta=5):
        if isinstance(queue_delta, bool) or not isinstance(queue_delta, int):
            raise TypeError("queue_delta must be an integer.")
        if queue_delta < 0:
            raise ValueError("queue_delta must be non-negative.")

        self.queue_delta = queue_delta

    def generate(self, current_belief):
        """Return {direction: hypothetical_belief} for all four directions.

        Each scenario starts from the same original belief. The four changes
        are independent; they are not cumulatively added to one another.
        Only the selected directional queue is changed. Waiting times, phase,
        vehicle count and observation timestamp are retained.
        """
        if current_belief is None or current_belief.state is None:
            raise ValueError("current_belief and its state are required.")

        state = current_belief.state
        scenarios = {}

        for direction in self.DIRECTIONS:
            queue_values = {
                "north_queue": state.north_queue,
                "south_queue": state.south_queue,
                "east_queue": state.east_queue,
                "west_queue": state.west_queue,
            }
            queue_field = f"{direction}_queue"
            queue_values[queue_field] += self.queue_delta

            hypothetical_state = type(state)(
                north_queue=queue_values["north_queue"],
                south_queue=queue_values["south_queue"],
                east_queue=queue_values["east_queue"],
                west_queue=queue_values["west_queue"],
                north_waiting=state.north_waiting,
                south_waiting=state.south_waiting,
                east_waiting=state.east_waiting,
                west_waiting=state.west_waiting,
                current_phase=state.current_phase,
                vehicle_count=state.vehicle_count,
            )

            scenarios[direction] = Belief(
                state=hypothetical_state,
                last_observation_time=current_belief.last_observation_time,
            )

        return scenarios

    def evaluate(self, current_belief, controller):
        """Evaluate action divergence for each independent directional change.

        Returns per-direction hypothetical actions and binary CVO values.
        multi_direction_cvo is 1 if at least one tested direction changes the
        selected action; otherwise it is 0. This is a sensitivity summary,
        not a probability that a real observation will change the action.
        """
        if controller is None:
            raise ValueError("controller is required.")

        current_action = controller.choose_action(current_belief)
        scenarios = self.generate(current_belief)
        directional_results = {}

        for direction, hypothetical_belief in scenarios.items():
            hypothetical_action = controller.choose_action(
                hypothetical_belief
            )
            directional_results[direction] = {
                "hypothetical_action": hypothetical_action,
                "cvo": action_divergence(
                    current_action, hypothetical_action
                ),
            }

        overall_cvo = max(
            result["cvo"] for result in directional_results.values()
        )

        return {
            "current_action": current_action,
            "directions": directional_results,
            "multi_direction_cvo": overall_cvo,
        }
