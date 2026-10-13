from src.cvo.action_divergence import action_divergence


# --------------------------------------------------
# DECISION MARGIN
# --------------------------------------------------

def calculate_decision_margin(belief):
    """
    Calculate the current controller decision margin.

    NS pressure = north queue + south queue
    EW pressure = east queue + west queue

    Margin = |NS pressure - EW pressure|
    """

    state = belief.state

    ns_pressure = (
        state.north_queue +
        state.south_queue
    )

    ew_pressure = (
        state.east_queue +
        state.west_queue
    )

    return abs(
        ns_pressure - ew_pressure
    )


# --------------------------------------------------
# CONTROL VALUE OF OBSERVATION
# --------------------------------------------------

def calculate_cvo(
    current_belief,
    hypothetical_belief,
    controller
):
    """
    Calculate CVO as action divergence between
    current and hypothetical controller actions.
    """

    # --------------------------------------------------
    # Current action
    # --------------------------------------------------

    current_action = controller.choose_action(
        current_belief
    )

    # --------------------------------------------------
    # Hypothetical action
    # --------------------------------------------------

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    # --------------------------------------------------
    # Action divergence
    # --------------------------------------------------

    cvo = action_divergence(
        current_action,
        hypothetical_action
    )

    return cvo


# --------------------------------------------------
# CVO CALCULATION + LOGGING
# --------------------------------------------------

def calculate_and_log_cvo(
    time,
    current_belief,
    hypothetical_belief,
    controller,
    logger
):
    """
    Calculate CVO and save the result using CVOLogger.
    """

    # --------------------------------------------------
    # Actions
    # --------------------------------------------------

    current_action = controller.choose_action(
        current_belief
    )

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    # --------------------------------------------------
    # CVO
    # --------------------------------------------------

    cvo = action_divergence(
        current_action,
        hypothetical_action
    )

    # --------------------------------------------------
    # Decision margin
    # --------------------------------------------------

    decision_margin = calculate_decision_margin(
        current_belief
    )

    # --------------------------------------------------
    # Belief representations
    # --------------------------------------------------

    current_state = current_belief.state
    hypothetical_state = hypothetical_belief.state

    current_belief_text = (
        f"NS="
        f"{current_state.north_queue + current_state.south_queue},"
        f"EW="
        f"{current_state.east_queue + current_state.west_queue}"
    )

    hypothetical_belief_text = (
        f"NS="
        f"{hypothetical_state.north_queue + hypothetical_state.south_queue},"
        f"EW="
        f"{hypothetical_state.east_queue + hypothetical_state.west_queue}"
    )

    # --------------------------------------------------
    # Log
    # --------------------------------------------------

    logger.log(
        time=time,
        current_belief=current_belief_text,
        hypothetical_belief=hypothetical_belief_text,
        current_action=current_action,
        hypothetical_action=hypothetical_action,
        decision_margin=decision_margin,
        cvo=cvo
    )

    return cvo