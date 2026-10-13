from src.acquisition.base_policy import AcquisitionPolicy


class AlwaysAcquirePolicy(AcquisitionPolicy):
    """
    Full-observation baseline.

    Acquire every available observation.
    """

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):
        """
        Always return True.

        True  -> ACQUIRE
        False -> SKIP
        """

        return True