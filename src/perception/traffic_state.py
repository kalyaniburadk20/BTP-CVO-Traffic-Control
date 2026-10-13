import traci


# --------------------------------------------------
# TRAFFIC STATE REPRESENTATION
# --------------------------------------------------

class TrafficState:
    """
    Represents the traffic condition at a particular
    simulation time.
    """

    def __init__(
        self,
        north_queue,
        south_queue,
        east_queue,
        west_queue,
        north_waiting,
        south_waiting,
        east_waiting,
        west_waiting,
        current_phase,
        vehicle_count
    ):
        self.north_queue = north_queue
        self.south_queue = south_queue
        self.east_queue = east_queue
        self.west_queue = west_queue

        self.north_waiting = north_waiting
        self.south_waiting = south_waiting
        self.east_waiting = east_waiting
        self.west_waiting = west_waiting

        self.current_phase = current_phase
        self.vehicle_count = vehicle_count

    def to_dict(self):
        """
        Convert TrafficState object to dictionary.
        Useful for CSV logging and debugging.
        """
        return {
            "vehicle_count": self.vehicle_count,

            "north_queue": self.north_queue,
            "south_queue": self.south_queue,
            "east_queue": self.east_queue,
            "west_queue": self.west_queue,

            "north_waiting": self.north_waiting,
            "south_waiting": self.south_waiting,
            "east_waiting": self.east_waiting,
            "west_waiting": self.west_waiting,

            "current_phase": self.current_phase,
        }


# --------------------------------------------------
# VEHICLE INFORMATION
# --------------------------------------------------

def get_vehicle_ids():
    """
    Return IDs of all vehicles currently
    active in the SUMO simulation.
    """
    return traci.vehicle.getIDList()


# --------------------------------------------------
# QUEUE INFORMATION
# --------------------------------------------------

def get_queue_lengths():
    """
    Count stopped/very slow vehicles on each
    incoming approach.
    """

    incoming_edges = {
        "north": "-E0",
        "south": "-E2",
        "east": "-E1",
        "west": "-E3",
    }

    queues = {
        "north_queue": 0,
        "south_queue": 0,
        "east_queue": 0,
        "west_queue": 0,
    }

    SPEED_THRESHOLD = 0.1

    for direction, edge_id in incoming_edges.items():

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

def get_waiting_times():
    """
    Calculate total waiting time of vehicles
    on each incoming approach.
    """

    incoming_edges = {
        "north": "-E0",
        "south": "-E2",
        "east": "-E1",
        "west": "-E3",
    }

    waiting_times = {
        "north_waiting": 0.0,
        "south_waiting": 0.0,
        "east_waiting": 0.0,
        "west_waiting": 0.0,
    }

    for direction, edge_id in incoming_edges.items():

        vehicle_ids = traci.edge.getLastStepVehicleIDs(
            edge_id
        )

        total_waiting = 0.0

        for vehicle_id in vehicle_ids:

            waiting_time = traci.vehicle.getWaitingTime(
                vehicle_id
            )

            total_waiting += waiting_time

        waiting_times[
            f"{direction}_waiting"
        ] = total_waiting

    return waiting_times


# --------------------------------------------------
# TRAFFIC SIGNAL
# --------------------------------------------------

def get_signal_phase():
    """
    Return the current traffic-light phase.
    """

    traffic_light_id = "J0"

    return traci.trafficlight.getPhase(
        traffic_light_id
    )


# --------------------------------------------------
# COMPLETE TRAFFIC STATE
# --------------------------------------------------

def get_traffic_state():
    """
    Read the current SUMO traffic information and
    return it as a TrafficState object.
    """

    vehicle_ids = get_vehicle_ids()

    queues = get_queue_lengths()

    waiting = get_waiting_times()

    phase = get_signal_phase()

    state = TrafficState(
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

    return state