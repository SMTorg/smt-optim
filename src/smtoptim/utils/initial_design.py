import numpy as np
import scipy.stats as stats

from smtoptim.core.state import OptimizationState


def generate_initial_design(state: OptimizationState, evaluator, seed=None) -> None:

    sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, seed=seed)
    doe = sampler.random(max(5, state.problem.num_dim))
    doe = stats.qmc.scale(doe,
                          state.problem.obj_configs[0].design_space[:, 0],
                          state.problem.obj_configs[0].design_space[:, 1])
    doe = [doe]
    infill = doe * state.problem.num_fidelity

    evaluator = evaluator(state.problem)
    evaluator.sample(infill, state)

