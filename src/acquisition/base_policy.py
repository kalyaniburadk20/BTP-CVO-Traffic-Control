class AcquisitionPolicy:
    """
    Common interface for all observation-acquisition policies.

    Every acquisition policy must implement:

        should_acquire(...)

    Return value:

        True  -> ACQUIRE
        False -> SKIP
    """

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):
        """
        Decide whether a new observation should be acquired.

        Parameters
        ----------
        current_time:
            Current simulation/control time.

        belief:
            Current belief available to the controller.

        observation_history:
            History of previously acquired observations.

        Returns
        -------
        bool
            True  -> ACQUIRE
            False -> SKIP
        """

        raise NotImplementedError(
            "Subclasses must implement should_acquire()."
        )