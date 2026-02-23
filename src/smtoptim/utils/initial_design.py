import copy

import numpy as np
import scipy.stats as stats

from smtoptim.core.state import OptimizationState


def generate_initial_design(state: OptimizationState, evaluator, config) -> None:


    if config.xt_init is None:

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, seed=config.seed)

        if config.nt_init is None:
            nt_init = max(5, state.problem.num_dim)
        else:
            nt_init = config.nt_init

        doe = sampler.random(nt_init)
        doe = stats.qmc.scale(doe,
                              state.problem.obj_configs[0].design_space[:, 0],
                              state.problem.obj_configs[0].design_space[:, 1])
        doe = [doe]
        infill = doe * state.problem.num_fidelity

    else:
        infill = copy.deepcopy(config.xt_init)

    evaluator.sample(infill, state)

