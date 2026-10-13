class HypotheticalObservationGenerator:
    """
    Generate a controlled hypothetical belief for CVO evaluation.

    This first implementation is deliberately deterministic.
    It does not use future SUMO ground truth.

    The hypothetical branch changes only the east-west queue
    information so that we can test whether the controller's
    action would change.
    """

    def __init__(self, east_queue_delta=5):

        if east_queue_delta < 0:
            raise ValueError(
                "east_queue_delta must be non-negative."
            )

        self.east_queue_delta = east_queue_delta

    def generate(self, current_belief):

        if current_belief is None:
            raise ValueError(
                "current_belief cannot be None."
            )

        if current_belief.state is None:
            raise ValueError(
                "current_belief.state cannot be None."
            )

        state = current_belief.state

        hypothetical_state = type(state)(
            north_queue=state.north_queue,
            south_queue=state.south_queue,
            east_queue=(
                state.east_queue
                + self.east_queue_delta
            ),
            west_queue=state.west_queue,
            north_waiting=state.north_waiting,
            south_waiting=state.south_waiting,
            east_waiting=state.east_waiting,
            west_waiting=state.west_waiting,
            current_phase=state.current_phase,
            vehicle_count=state.vehicle_count
        )

        from src.perception.belief import Belief

        return Belief(
            state=hypothetical_state,
            last_observation_time=(
                current_belief.last_observation_time
            )
        )