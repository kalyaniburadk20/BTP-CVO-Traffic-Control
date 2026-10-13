# --------------------------------------------------
# BELIEF REPRESENTATION
# --------------------------------------------------

class Belief:
    """
    Represents the traffic-state information currently
    available to the controller.
    """

    def __init__(self, state=None, last_observation_time=None):
        self.state = state
        self.last_observation_time = last_observation_time

    def update(self, observation):
        """
        Update the belief using a newly acquired observation.
        """

        self.state = observation.traffic_state
        self.last_observation_time = observation.timestamp

    def retain(self):
        """
        Retain the current belief.

        Used when an observation is skipped.
        """
        pass

    def get_age(self, current_time):
        """
        Return the age of the latest observation.
        """

        if self.last_observation_time is None:
            return None

        return current_time - self.last_observation_time