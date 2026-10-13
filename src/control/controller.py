# --------------------------------------------------
# SIGNAL CONTROLLER
# --------------------------------------------------

class SignalController:
    """
    Deterministic queue-pressure based signal controller.

    Input:
        Belief

    Output:
        Desired signal action
    """

    NS_GREEN = "NS_GREEN"
    EW_GREEN = "EW_GREEN"

    def choose_action(self, belief):
        """
        Choose the desired signal action from the
        current traffic belief.
        """

        state = belief.state

        # Calculate pressure for both directions
        ns_pressure = (
            state.north_queue
            + state.south_queue
        )

        ew_pressure = (
            state.east_queue
            + state.west_queue
        )

        # NS has higher pressure
        if ns_pressure > ew_pressure:
            return self.NS_GREEN

        # EW has higher pressure
        if ew_pressure > ns_pressure:
            return self.EW_GREEN

        # Tie:
        # keep the currently preferred action
        return self._get_current_action(belief)

    def _get_current_action(self, belief):
        """
        Determine the current preferred action.

        For the initial implementation, use the current
        traffic-light phase stored in the belief.

        Phase mapping:
            0, 1 -> NS direction
            2, 3 -> EW direction
        """

        phase = belief.state.current_phase

        if phase in (0, 1):
            return self.NS_GREEN

        return self.EW_GREEN