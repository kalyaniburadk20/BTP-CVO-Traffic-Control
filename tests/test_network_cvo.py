import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.perception.network_traffic_state import (
    get_intersection_traffic_state
)

from src.perception.observation import Observation
from src.perception.belief import Belief

from src.control.network_controller import (
    NetworkSignalController
)

from src.cvo.cvo import calculate_cvo
from src.cvo.hypothetical import (
    HypotheticalObservationGenerator
)


SUMO_CONFIG = r".\sumo\configs\low_network.sumocfg"

INTERSECTIONS = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5"
]


traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG
])

try:

    beliefs = {
        intersection_id: Belief()
        for intersection_id in INTERSECTIONS
    }

    controllers = {
        intersection_id:
            NetworkSignalController(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    hypothetical_generator = (
        HypotheticalObservationGenerator(
            east_queue_delta=5
        )
    )

    cvo_counts = {
        intersection_id: {
            0.0: 0,
            1.0: 0
        }
        for intersection_id in INTERSECTIONS
    }

    # Bootstrap
    traci.simulationStep()

    current_time = traci.simulation.getTime()

    for intersection_id in INTERSECTIONS:

        state = get_intersection_traffic_state(
            intersection_id
        )

        observation = Observation(
            traffic_state=state,
            timestamp=current_time
        )

        beliefs[intersection_id].update(
            observation
        )

    # Evaluate CVO for the next 60 steps
        # Evaluate CVO for the next 60 steps
    for _ in range(60):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        for intersection_id in INTERSECTIONS:

            # Read the actual current SUMO state
            state = get_intersection_traffic_state(
                intersection_id
            )

            observation = Observation(
                traffic_state=state,
                timestamp=current_time
            )

            # Update belief with current observation
            beliefs[intersection_id].update(
                observation
            )

            # Generate hypothetical belief
            hypothetical_belief = (
                hypothetical_generator.generate(
                    beliefs[intersection_id]
                )
            )

            # Calculate CVO
            cvo = calculate_cvo(
                current_belief=beliefs[intersection_id],
                hypothetical_belief=hypothetical_belief,
                controller=controllers[intersection_id]
            )

            cvo_counts[
                intersection_id
            ][cvo] += 1

        

        
            

            

            

    print("Network CVO test completed.")
    print()

    for intersection_id in INTERSECTIONS:

        print(
            intersection_id,
            "| CVO=0:",
            cvo_counts[intersection_id][0.0],
            "| CVO=1:",
            cvo_counts[intersection_id][1.0]
        )

finally:

    traci.close()