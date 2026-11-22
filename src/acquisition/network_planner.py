# --------------------------------------------------
# NETWORK OBSERVATION PLANNER
# --------------------------------------------------

class NetworkObservationPlanner:
    """
    Centralized planner for selecting which intersections
    should acquire observations.

    Selection is based on:
        1. Forced refresh for beliefs at/above max_belief_age
        2. CVO > observation cost
        3. Higher CVO first
        4. Optional belief-age tie-break before decision margin
        5. Network-wide observation budget

    age_priority_on_ties defaults to False to preserve the original
    planner's selection behavior. Experimental policies can enable it
    to reduce repeated preference for the same high-margin intersection.
    """

    def __init__(
        self,
        observation_budget=2,
        max_belief_age=10.0,
        age_priority_on_ties=False,
    ):
        if observation_budget <= 0:
            raise ValueError("observation_budget must be positive.")

        if max_belief_age < 0:
            raise ValueError("max_belief_age cannot be negative.")

        if not isinstance(age_priority_on_ties, bool):
            raise TypeError("age_priority_on_ties must be a boolean.")

        self.observation_budget = observation_budget
        self.max_belief_age = max_belief_age
        self.age_priority_on_ties = age_priority_on_ties
        # Remember the last forced-refresh selection so equal-age stale
        # intersections are served in round-robin order.
        self._last_forced_selection = None

    def select_intersections(
        self,
        cvo_values,
        cost_values,
        decision_margins,
        belief_ages=None,
    ):
        """
        Select intersections for observation.

        Parameters
        ----------
        cvo_values : dict
            {intersection_id: normalized CVO score}

        cost_values : dict
            {intersection_id: total normalized observation cost}

        decision_margins : dict
            {intersection_id: decision margin}

        belief_ages : dict, optional
            {intersection_id: age of the current belief}

        Returns
        -------
        list
            Selected intersection IDs, up to observation_budget.
        """

        intersection_ids = set(cvo_values.keys())

        # Older callers may not provide belief ages.
        # Without age information, assume no forced refresh.
        if belief_ages is None:
            belief_ages = {
                intersection_id: 0.0
                for intersection_id in intersection_ids
            }

        if intersection_ids != set(cost_values.keys()):
            raise ValueError(
                "cvo_values and cost_values must contain the same intersections."
            )

        if intersection_ids != set(decision_margins.keys()):
            raise ValueError(
                "cvo_values and decision_margins must contain the same intersections."
            )

        if intersection_ids != set(belief_ages.keys()):
            raise ValueError(
                "cvo_values and belief_ages must contain the same intersections."
            )

        forced = []
        eligible = []

        for intersection_id in intersection_ids:
            cvo = cvo_values[intersection_id]
            cost = cost_values[intersection_id]
            margin = decision_margins[intersection_id]
            age = belief_ages[intersection_id]

            if cvo < 0 or cvo > 1:
                raise ValueError("CVO must be between 0 and 1.")

            if cost < 0:
                raise ValueError("Observation cost cannot be negative.")

            if margin < 0:
                raise ValueError("Decision margin cannot be negative.")

            if age < 0:
                raise ValueError("Belief age cannot be negative.")

            # Stale beliefs are prioritized before CVO-eligible candidates.
            if age >= self.max_belief_age:
                forced.append((intersection_id, age))
            elif cvo > cost:
                eligible.append((intersection_id, cvo, age, margin))

        # Among forced refreshes, serve the oldest beliefs first. When ages
        # tie, rotate the order after the last forced selection so that a
        # low intersection ID is not always selected first.
        ordered_ids = sorted(intersection_ids)
        if self._last_forced_selection in ordered_ids:
            start_index = (
                ordered_ids.index(self._last_forced_selection) + 1
            ) % len(ordered_ids)
        else:
            start_index = 0

        cyclic_rank = {
            intersection_id: (
                ordered_ids.index(intersection_id) - start_index
            ) % len(ordered_ids)
            for intersection_id in ordered_ids
        }
        forced.sort(
            key=lambda item: (-item[1], cyclic_rank[item[0]])
        )

        if self.age_priority_on_ties:
            # CVO remains the primary criterion. If CVO is tied, refresh
            # the older belief first, then use decision margin.
            eligible.sort(
                key=lambda item: (item[1], item[2], item[3]),
                reverse=True,
            )
        else:
            # Preserve the original CVO-then-margin selection order.
            eligible.sort(
                key=lambda item: (item[1], item[3]),
                reverse=True,
            )

        selected = []

        selected_forced = []
        for intersection_id, age in forced:
            if len(selected) >= self.observation_budget:
                break
            selected.append(intersection_id)
            selected_forced.append(intersection_id)

        if selected_forced:
            self._last_forced_selection = selected_forced[-1]

        for intersection_id, cvo, age, margin in eligible:
            if len(selected) >= self.observation_budget:
                break
            if intersection_id not in selected:
                selected.append(intersection_id)

        return selected
