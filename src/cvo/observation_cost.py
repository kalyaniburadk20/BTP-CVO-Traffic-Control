# --------------------------------------------------
# OBSERVATION COST MODEL
# --------------------------------------------------


class ObservationCost:
    """
    Observation cost model containing:
    - compute cost
    - communication cost
    - latency cost

    Raw values are kept conceptually separate from
    their normalized values.

    Initial default weights are equal:
        compute       = 1/3
        communication = 1/3
        latency       = 1/3
    """

    def __init__(
        self,
        compute_weight=1 / 3,
        communication_weight=1 / 3,
        latency_weight=1 / 3
    ):

        # --------------------------------------------------
        # Validate weights
        # --------------------------------------------------

        if compute_weight < 0:
            raise ValueError(
                "compute_weight cannot be negative."
            )

        if communication_weight < 0:
            raise ValueError(
                "communication_weight cannot be negative."
            )

        if latency_weight < 0:
            raise ValueError(
                "latency_weight cannot be negative."
            )

        total_weight = (
            compute_weight
            + communication_weight
            + latency_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one cost weight must be positive."
            )

        # --------------------------------------------------
        # Normalize weights so they sum to 1
        # --------------------------------------------------

        self.compute_weight = (
            compute_weight / total_weight
        )

        self.communication_weight = (
            communication_weight / total_weight
        )

        self.latency_weight = (
            latency_weight / total_weight
        )

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def normalize_cost(
        raw_cost,
        reference_cost
    ):
        """
        Normalize a raw cost using:

            normalized = raw / reference

        Result is clipped to [0, 1].
        """

        if raw_cost < 0:
            raise ValueError(
                "raw_cost cannot be negative."
            )

        if reference_cost <= 0:
            raise ValueError(
                "reference_cost must be greater than 0."
            )

        normalized = (
            raw_cost / reference_cost
        )

        return min(
            normalized,
            1.0
        )

    # --------------------------------------------------
    # TOTAL COST
    # --------------------------------------------------

    def calculate(
        self,
        compute_cost,
        communication_cost,
        latency_cost,
        compute_reference,
        communication_reference,
        latency_reference
    ):
        """
        Calculate normalized and weighted total
        observation cost.

        Raw values are returned as well so that later
        experiments can preserve both raw and normalized
        measurements.
        """

        # --------------------------------------------------
        # Normalize each component
        # --------------------------------------------------

        normalized_compute = self.normalize_cost(
            compute_cost,
            compute_reference
        )

        normalized_communication = self.normalize_cost(
            communication_cost,
            communication_reference
        )

        normalized_latency = self.normalize_cost(
            latency_cost,
            latency_reference
        )

        # --------------------------------------------------
        # Weighted total
        # --------------------------------------------------

        total_cost = (
            self.compute_weight
            * normalized_compute
            +
            self.communication_weight
            * normalized_communication
            +
            self.latency_weight
            * normalized_latency
        )

        # --------------------------------------------------
        # Return complete information
        # --------------------------------------------------

        return {
            "raw_compute": compute_cost,
            "raw_communication": communication_cost,
            "raw_latency": latency_cost,

            "normalized_compute":
                normalized_compute,

            "normalized_communication":
                normalized_communication,

            "normalized_latency":
                normalized_latency,

            "compute_weight":
                self.compute_weight,

            "communication_weight":
                self.communication_weight,

            "latency_weight":
                self.latency_weight,

            "total_cost":
                total_cost
        }