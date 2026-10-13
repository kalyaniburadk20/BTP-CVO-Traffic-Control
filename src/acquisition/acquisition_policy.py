# --------------------------------------------------
# CVO ACQUISITION POLICY
# --------------------------------------------------


class CVOAcquisitionPolicy:
    """
    Acquisition policy based on the comparison between
    Control Value of Observation (CVO) and total
    observation cost.
    """

    def should_acquire(
        self,
        cvo,
        total_cost
    ):
        """
        Decide whether a new observation should be acquired.

        Rule:

            ACQUIRE if CVO > total_cost
            SKIP otherwise
        """

        if cvo < 0:
            raise ValueError(
                "CVO cannot be negative."
            )

        if total_cost < 0:
            raise ValueError(
                "total_cost cannot be negative."
            )

        return cvo > total_cost