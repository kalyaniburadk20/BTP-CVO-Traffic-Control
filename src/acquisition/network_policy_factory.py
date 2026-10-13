# --------------------------------------------------
# NETWORK ACQUISITION POLICY FACTORY
# --------------------------------------------------

from src.acquisition.always_acquire import (
    AlwaysAcquirePolicy
)

from src.acquisition.periodic import (
    PeriodicPolicy
)

from src.acquisition.network_state_change import (
    NetworkStateChangePolicy
)
from src.acquisition.aoi import (
    AoIPolicy
)


def create_network_policies(
    strategy,
    intersection_ids,
    config
):
    """
    Create one independent acquisition-policy
    instance for every network intersection.
    """

    if strategy not in (
        "always",
        "periodic",
        "state_change",
        "aoi"
    ):
        raise ValueError(
            f"Unsupported baseline strategy: {strategy}"
        )

    policies = {}

    for intersection_id in intersection_ids:

        if strategy == "always":

            policy = AlwaysAcquirePolicy()

        elif strategy == "periodic":

            policy = PeriodicPolicy(
                interval=config[
                    "acquisition"
                ][
                    "periodic_interval"
                ]
            )

        elif strategy == "state_change":

            policy = NetworkStateChangePolicy(
                threshold=config[
                    "acquisition"
                ][
                    "state_change_threshold"
                ]
            )

        else:

            policy = AoIPolicy(
                max_age=config[
                    "acquisition"
                ][
                    "aoi_max_age"
                ]
            )

        policies[intersection_id] = policy

    return policies