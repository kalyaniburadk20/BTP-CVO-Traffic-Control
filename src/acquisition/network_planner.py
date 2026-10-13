# --------------------------------------------------
# NETWORK OBSERVATION PLANNER
# --------------------------------------------------

class NetworkObservationPlanner:
    """
    Centralized planner for selecting which
    intersections should acquire observations.

    Selection is based on:
        1. CVO > observation cost
        2. Higher CVO first
        3. Decision margin as tie-breaker
        4. Network-wide observation budget
    """

    def __init__(
        self,
        observation_budget=2,
        max_belief_age=10.0
    ):
        if observation_budget <= 0:
            raise ValueError(
                "observation_budget must be positive."
            )

        self.observation_budget = observation_budget
        self.max_belief_age = max_belief_age

    def select_intersections(
        self,
        cvo_values,
        cost_values,
        decision_margins,
        belief_ages=None
    ):
        """
        Select intersections for observation.

        Parameters
        ----------
        cvo_values : dict
            {intersection_id: cvo}

        cost_values : dict
            {intersection_id: total_cost}

        decision_margins : dict
            {intersection_id: decision_margin}

        belief_ages : dict
            {intersection_id: belief_age}

        Returns
        -------
        list
            Selected intersection IDs.
        """

        intersection_ids = set(
            cvo_values.keys()
        )

                # Older callers may not provide belief ages.
        # Without age information, assume no forced refresh.
        if belief_ages is None:
            belief_ages = {
                intersection_id: 0.0
                for intersection_id in intersection_ids
            }

        if intersection_ids != set(
            cost_values.keys()
        ):
            raise ValueError(
                "cvo_values and cost_values "
                "must contain the same intersections."
            )

        if intersection_ids != set(
            decision_margins.keys()
        ):
            raise ValueError(
                "cvo_values and decision_margins "
                "must contain the same intersections."
            )

        if intersection_ids != set(
            belief_ages.keys()
        ):
            raise ValueError(
                "cvo_values and belief_ages "
                "must contain the same intersections."
            )

        forced = []
        eligible = []

        for intersection_id in intersection_ids:
            cvo = cvo_values[intersection_id]
            cost = cost_values[intersection_id]
            margin = decision_margins[intersection_id]
            age = belief_ages[intersection_id]

            if cvo < 0:
                raise ValueError("CVO cannot be negative.")

            if cost < 0:
                raise ValueError("Observation cost cannot be negative.")

            if margin < 0:
                raise ValueError("Decision margin cannot be negative.")

            if age < 0:
                raise ValueError("Belief age cannot be negative.")

            # Prioritize stale beliefs before CVO-eligible intersections.
            if age >= self.max_belief_age:
                forced.append((intersection_id, age))
            elif cvo > cost:
                eligible.append((intersection_id, cvo, margin))

        forced.sort(key=lambda item: (-item[1], item[0]))
        eligible.sort(
            key=lambda item: (item[1], item[2]),
            reverse=True
        )

        selected = []

        for intersection_id, age in forced:
            if len(selected) >= self.observation_budget:
                break
            selected.append(intersection_id)

        for intersection_id, cvo, margin in eligible:
            if len(selected) >= self.observation_budget:
                break
            if intersection_id not in selected:
                selected.append(intersection_id)

        return selected