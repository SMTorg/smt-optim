


def check_stop_criteria(state, config) -> bool:

    if state.iter >= config.max_iter:
        return False

    elif state.budget > config.max_budget:
        return False

    else:
        return True

