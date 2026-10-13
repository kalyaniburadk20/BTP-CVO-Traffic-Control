# --------------------------------------------------
# NETWORK SIGNAL PHASE CONFIGURATION
# --------------------------------------------------

INTERSECTION_SIGNAL_CONFIG = {

    "J1": {
        "NS_GREEN_PHASE": 4,
        "EW_GREEN_PHASE": 0,
        "NS_YELLOW_PHASE": 5,
        "EW_YELLOW_PHASE": 1,
    },

    "J2": {
        "NS_GREEN_PHASE": 2,
        "EW_GREEN_PHASE": 0,
        "NS_YELLOW_PHASE": 3,
        "EW_YELLOW_PHASE": 1,
    },

    "J3": {
        "NS_GREEN_PHASE": 4,
        "EW_GREEN_PHASE": 0,
        "NS_YELLOW_PHASE": 5,
        "EW_YELLOW_PHASE": 1,
    },

    "J4": {
        "NS_GREEN_PHASE": 0,
        "EW_GREEN_PHASE": 2,
        "NS_YELLOW_PHASE": 1,
        "EW_YELLOW_PHASE": 3,
    },

    "J5": {
        "NS_GREEN_PHASE": 0,
        "EW_GREEN_PHASE": 2,
        "NS_YELLOW_PHASE": 1,
        "EW_YELLOW_PHASE": 3,
    },
}


def get_signal_config(intersection_id):

    if intersection_id not in INTERSECTION_SIGNAL_CONFIG:
        raise ValueError(
            f"Unknown intersection: {intersection_id}"
        )

    return INTERSECTION_SIGNAL_CONFIG[
        intersection_id
    ]