import traci

from src.perception.traffic_state import TrafficState


# --------------------------------------------------
# NETWORK INTERSECTION CONFIGURATION
# --------------------------------------------------

INTERSECTION_EDGES = {

    "J1": {
        "north": "N1_to_J1",
        "east": "J2_to_J1",
        "south": "J4_to_J1",
        "west": "W1_to_J1",
    },

    "J2": {
        "north": "N2_to_J2",
        "east": "J3_to_J2",
        "south": "J5_to_J2",
        "west": "J1_to_J2",
    },

    "J3": {
        "north": "N3_to_J3",
        "east": "E3_to_J3",
        "south": "S3_to_J3",
        "west": "J2_to_J3",
    },

    "J4": {
        "north": "J1_to_J4",
        "east": "J5_to_J4",
        "south": "S4_to_J4",
        "west": "W4_to_J4",
    },

    "J5": {
        "north": "J2_to_J5",
        "east": "E5_to_J5",
        "south": "S5_to_J5",
        "west": "J4_to_J5",
    },
}


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_intersection_id(intersection_id):
    if intersection_id not in INTERSECTION_EDGES:
        raise ValueError(
            f"Unknown intersection: {intersection_id}"
        )


# --------------------------------------------------
# QUEUE INFORMATION
# --------------------------------------------------

def get_intersection_queue_lengths(intersection_id):
    validate_intersection_id(intersection_id)

    edge_map = INTERSECTION_EDGES[intersection_id]

    queues = {
        "north_queue": 0,
        "south_queue": 0,
        "east_queue": 0,
        "west_queue": 0,
    }

    SPEED_THRESHOLD = 0.1

    for direction, edge_id in edge_map.items():

        vehicle_ids = traci.edge.getLastStepVehicleIDs(
            edge_id
        )

        queue_count = 0

        for vehicle_id in vehicle_ids:

            speed = traci.vehicle.getSpeed(
                vehicle_id
            )

            if speed <= SPEED_THRESHOLD:
                queue_count += 1

        queues[f"{direction}_queue"] = queue_count

    return queues


# --------------------------------------------------
# WAITING TIME
# --------------------------------------------------

def get_intersection_waiting_times(intersection_id):
    validate_intersection_id(intersection_id)

    edge_map = INTERSECTION_EDGES[intersection_id]

    waiting_times = {
        "north_waiting": 0.0,
        "south_waiting": 0.0,
        "east_waiting": 0.0,
        "west_waiting": 0.0,
    }

    for direction, edge_id in edge_map.items():

        vehicle_ids = traci.edge.getLastStepVehicleIDs(
            edge_id
        )

        total_waiting = 0.0

        for vehicle_id in vehicle_ids:

            total_waiting += traci.vehicle.getWaitingTime(
                vehicle_id
            )

        waiting_times[
            f"{direction}_waiting"
        ] = total_waiting

    return waiting_times


# --------------------------------------------------
# SIGNAL PHASE
# --------------------------------------------------

def get_intersection_signal_phase(intersection_id):
    validate_intersection_id(intersection_id)

    return traci.trafficlight.getPhase(
        intersection_id
    )


# --------------------------------------------------
# COMPLETE INTERSECTION STATE
# --------------------------------------------------

def get_intersection_traffic_state(intersection_id):
    validate_intersection_id(intersection_id)

    vehicle_ids = traci.vehicle.getIDList()

    queues = get_intersection_queue_lengths(
        intersection_id
    )

    waiting = get_intersection_waiting_times(
        intersection_id
    )

    phase = get_intersection_signal_phase(
        intersection_id
    )

    return TrafficState(
        north_queue=queues["north_queue"],
        south_queue=queues["south_queue"],
        east_queue=queues["east_queue"],
        west_queue=queues["west_queue"],

        north_waiting=waiting["north_waiting"],
        south_waiting=waiting["south_waiting"],
        east_waiting=waiting["east_waiting"],
        west_waiting=waiting["west_waiting"],

        current_phase=phase,

        vehicle_count=len(vehicle_ids)
    )