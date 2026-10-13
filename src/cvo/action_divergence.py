# --------------------------------------------------
# ACTION DIVERGENCE
# --------------------------------------------------

def action_divergence(current_action, hypothetical_action):
    """
    Measure whether the hypothetical action differs
    from the current action.

    Returns:
        1.0 -> actions are different
        0.0 -> actions are the same
    """

    if current_action != hypothetical_action:
        return 1.0

    return 0.0