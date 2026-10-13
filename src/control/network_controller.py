# --------------------------------------------------
# NETWORK SIGNAL CONTROLLER
# --------------------------------------------------

from src.control.network_signal_config import (
    get_signal_config
)


class NetworkSignalController:
    """
    Queue-pressure based controller for one
    intersection in the multi-intersection network.
    """

    NS_GREEN = "NS_GREEN"
    EW_GREEN = "EW_GREEN"

    def __init__(self, intersection_id):

        self.intersection_id = intersection_id

        # Validate intersection configuration
        get_signal_config(intersection_id)

    def choose_action(self, belief):
        """
        Choose the desired signal action from
        the belief of this intersection.
        """

        if belief is None:
            raise ValueError(
                "belief cannot be None."
            )

        if belief.state is None:
            raise ValueError(
                "belief.state cannot be None."
            )

        state = belief.state

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

        # Tie: use current signal phase
        return self._get_current_action(
            state.current_phase
        )

    def _get_current_action(self, phase):
        """
        Determine the current preferred action
        from the configured green phases.
        """

        config = get_signal_config(
            self.intersection_id
        )

        if phase == config["NS_GREEN_PHASE"]:
            return self.NS_GREEN

        if phase == config["EW_GREEN_PHASE"]:
            return self.EW_GREEN

        # During yellow/special transition phases,
        # use the currently active side based on
        # the configured phase structure.
        if self.intersection_id in ("J1", "J3"):

            if phase in (1, 2, 3):
                return self.EW_GREEN

        if phase == config["NS_YELLOW_PHASE"]:
            return self.NS_GREEN

        if phase == config["EW_YELLOW_PHASE"]:
            return self.EW_GREEN

        raise ValueError(
            f"Unknown signal phase {phase} "
            f"for intersection "
            f"{self.intersection_id}."
        )