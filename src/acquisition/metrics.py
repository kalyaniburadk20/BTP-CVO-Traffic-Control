# --------------------------------------------------
# ACQUISITION RATE
# --------------------------------------------------

def calculate_acquisition_rate(acquired_count, available_count):
    if available_count < 0:
        raise ValueError("available_count cannot be negative.")

    if acquired_count < 0:
        raise ValueError("acquired_count cannot be negative.")

    if acquired_count > available_count:
        raise ValueError(
            "acquired_count cannot exceed available_count."
        )

    if available_count == 0:
        return 0.0

    return acquired_count / available_count


def calculate_acquisition_percentage(
    acquired_count,
    available_count
):
    rate = calculate_acquisition_rate(
        acquired_count,
        available_count
    )

    return rate * 100.0


# --------------------------------------------------
# COMMUNICATION COST
# --------------------------------------------------

def calculate_communication_data(
    acquired_count,
    observation_size_bytes
):
    if acquired_count < 0:
        raise ValueError(
            "acquired_count cannot be negative."
        )

    if observation_size_bytes < 0:
        raise ValueError(
            "observation_size_bytes cannot be negative."
        )

    return acquired_count * observation_size_bytes


# --------------------------------------------------
# COMPUTE COST
# --------------------------------------------------

def calculate_compute_calls(acquired_count):
    if acquired_count < 0:
        raise ValueError(
            "acquired_count cannot be negative."
        )

    return acquired_count


# --------------------------------------------------
# COMPUTE SAVINGS
# --------------------------------------------------

def calculate_compute_savings(
    compute_always,
    compute_policy
):
    if compute_always <= 0:
        raise ValueError(
            "compute_always must be greater than 0."
        )

    if compute_policy < 0:
        raise ValueError(
            "compute_policy cannot be negative."
        )

    return 1.0 - (
        compute_policy / compute_always
    )


# --------------------------------------------------
# LATENCY RECORD
# --------------------------------------------------

def create_latency_record(
    acquisition_time,
    processing_time,
    controller_decision_time
):
    return {
        "acquisition_time": acquisition_time,
        "processing_time": processing_time,
        "controller_decision_time":
            controller_decision_time
    }