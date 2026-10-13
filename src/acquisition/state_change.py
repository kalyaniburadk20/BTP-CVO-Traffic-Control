from src.acquisition.base_policy import AcquisitionPolicy
from src.acquisition.config_loader import load_acquisition_config


class StateChangePolicy(AcquisitionPolicy):
    """
    Traffic-aware acquisition policy.

    A new observation is acquired when the queue-based
    distance from the last acquired state exceeds the
    configured threshold.
    """

    def __init__(self, threshold=None):

        # If no threshold is provided,
        # load it from acquisition.yaml.
        if threshold is None:
            config = load_acquisition_config()

            threshold = config["state_change"]["threshold"]

        if threshold < 0:
            raise ValueError(
                "Threshold must be non-negative."
            )

        self.threshold = threshold
        self.last_acquired_state = None

    def _get_queue_state(self, state):
        """
        Extract only queue lengths from a TrafficState.
        """

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
        """
        Calculate L1 distance between current and
        last acquired queue states.
        """

        current_queues = self._get_queue_state(
            current_state
        )

        last_queues = self._get_queue_state(
            last_state
        )

        distance = sum(
            abs(current - last)
            for current, last in zip(
                current_queues,
                last_queues
            )
        )

        return distance

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):
        """
        First observation:
            ACQUIRE

        Later:
            distance > threshold -> ACQUIRE
            distance <= threshold -> SKIP
        """

        current_state = belief.state

        # First observation
        if self.last_acquired_state is None:

            self.last_acquired_state = current_state

            return True

        # Calculate distance from last acquired state
        distance = self._calculate_distance(
            current_state,
            self.last_acquired_state
        )

        # Sufficient state change
        if distance > self.threshold:

            self.last_acquired_state = current_state

            return True

        # Insufficient state change
        return False