import numpy as np


def check_stop_criteria(state, config) -> bool:
    # state.budget = compute_budget(state)

    if state.iter >= config.max_iter:
        return False

    elif state.budget >= config.max_budget:
        return False
    else:
        return True


def compute_budget(state) -> float:

    costs = state.problem.costs
    data = state.dataset.export_as_dict()

    masks = [(data["fidelity"] == lvl).ravel() for lvl in range(state.problem.num_fidelity)]

    budget = 0
    for lvl in range(state.problem.num_fidelity):
        budget += np.count_nonzero(masks[lvl]) * costs[lvl]

    return budget