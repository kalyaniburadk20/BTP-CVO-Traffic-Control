from src.acquisition.base_policy import AcquisitionPolicy
from src.acquisition.config_loader import load_acquisition_config


class AoIPolicy(AcquisitionPolicy):
    """
    Age of Information (AoI) acquisition policy.

    Acquire a new observation when the information
    has become sufficiently old.
    """

    def __init__(self, max_age=None):

        # If no max_age is provided,
        # load it from acquisition.yaml.
        if max_age is None:
            config = load_acquisition_config()

            max_age = config["aoi"]["max_age"]

        if max_age <= 0:
            raise ValueError(
                "max_age must be greater than 0."
            )

        self.max_age = max_age

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):
        """
        First observation:
            ACQUIRE

        Otherwise:
            age >= max_age -> ACQUIRE
            age < max_age   -> SKIP
        """

        # No belief exists yet.
        if belief is None:
            return True

        last_observation_time = (
            belief.last_observation_time
        )

        # No previous observation exists.
        if last_observation_time is None:
            return True

        # Calculate Age of Information.
        age = (
            current_time
            - last_observation_time
        )

        return age >= self.max_age