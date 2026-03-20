import copy

import numpy as np
from smt.sampling_methods import LHS
from smt.applications import NestedLHS

from smt_optim.core.state import State


def generate_initial_design(state: State, evaluator, config) -> None:


    if config.xt_init is None:

        if state.problem.num_fidelity == 1:
            sampler = LHS(xlimits=state.problem.design_space,
                          criterion="ese",
                          seed=config.seed)
        else:
            sampler = NestedLHS(xlimits=state.problem.design_space,
                                nlevel=state.problem.num_fidelity,
                                seed=config.seed)

        if config.nt_init is None:
            nt_init = max(5, state.problem.num_dim+1)
        else:
            nt_init = config.nt_init

        doe = sampler(nt_init)

        if state.problem.num_fidelity == 1:
            doe = [doe]
            infill = doe * state.problem.num_fidelity
        else:
            infill = doe

    else:
        infill = copy.deepcopy(config.xt_init)

    evaluator.sample_func(infill, state)

