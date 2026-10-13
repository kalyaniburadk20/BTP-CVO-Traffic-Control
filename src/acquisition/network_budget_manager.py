# --------------------------------------------------
# NETWORK OBSERVATION BUDGET MANAGER
# --------------------------------------------------

class NetworkBudgetManager:
    """
    Enforces a common network-wide observation budget.

    Each intersection first makes a local request.
    The budget manager then selects at most
    observation_budget intersections.

    A rotating pointer provides deterministic
    and fair tie-breaking when multiple intersections
    request observation at the same time.
    """

    def __init__(self, observation_budget, intersection_ids):

        if observation_budget <= 0:
            raise ValueError(
                "observation_budget must be positive."
            )

        if not intersection_ids:
            raise ValueError(
                "intersection_ids cannot be empty."
            )

        self.observation_budget = (
            observation_budget
        )

        self.intersection_ids = list(
            intersection_ids
        )

        self.pointer = 0

    def select_requested(
        self,
        requests,
        forced_intersections=None
    ):
        """
        Parameters
        ----------
        requests : dict
            {intersection_id: True/False}

        Returns
        -------
        list
            Selected intersections.
        """

        if set(requests.keys()) != set(
            self.intersection_ids
        ):
            raise ValueError(
                "requests must contain exactly "
                "the configured intersections."
            )

        requested = [
            intersection_id
            for intersection_id in self.intersection_ids
            if requests[intersection_id]
        ]

        # --------------------------------------------------
        # FORCED FRESHNESS
        # --------------------------------------------------
        forced_intersections = forced_intersections or []
        forced = []

        for intersection_id in forced_intersections:

            if intersection_id not in forced:
                forced.append(
                    intersection_id
                )

        selected = forced[
            :self.observation_budget
        ]

        # --------------------------------------------------
        # FILL REMAINING BUDGET
        # --------------------------------------------------

        remaining_budget = (
            self.observation_budget
            - len(selected)
        )

        if remaining_budget > 0:

            n = len(
                self.intersection_ids
            )

            ordered_ids = [
                self.intersection_ids[
                    (self.pointer + offset) % n
                ]
                for offset in range(n)
            ]

            ordered_requested = [
                intersection_id
                for intersection_id in ordered_ids
                if (
                    intersection_id in requested
                    and intersection_id not in selected
                )
            ]

            selected.extend(
                ordered_requested[
                    :remaining_budget
                ]
            )

        # --------------------------------------------------
        # UPDATE ROTATING POINTER
        # --------------------------------------------------

        self.pointer = (
            self.pointer + len(selected)
        ) % len(self.intersection_ids)

        return selected