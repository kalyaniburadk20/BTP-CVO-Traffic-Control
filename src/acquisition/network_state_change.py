from src.acquisition.base_policy import AcquisitionPolicy


class NetworkStateChangePolicy(AcquisitionPolicy):
    """
    Network-safe state-change policy.

    A request does not update the reference state.
    The reference state is updated only after the
    observation is actually acquired.
    """

    def __init__(self, threshold=10):

        if threshold < 0:
            raise ValueError(
                "Threshold must be non-negative."
            )

        self.threshold = threshold
        self.last_acquired_state = None

    def _get_queue_state(self, state):

        return (
            state.north_queue,
            state.south_queue,
            state.east_queue,
            state.west_queue
        )

    def _calculate_distance(
        self,
        current_state,
        last_state
    ):

        current_queues = self._get_queue_state(
            current_state
        )

        last_queues = self._get_queue_state(
            last_state
        )

        return sum(
            abs(current - last)
            for current, last in zip(
                current_queues,
                last_queues
            )
        )

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):

        current_state = belief.state

        if current_state is None:
            return False

        if self.last_acquired_state is None:
            return True

        distance = self._calculate_distance(
            current_state,
            self.last_acquired_state
        )

        return distance > self.threshold

    def record_acquisition(self, state):

        if state is None:
            raise ValueError(
                "state cannot be None."
            )

        self.last_acquired_state = state