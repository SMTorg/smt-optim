import copy

import numpy as np
from smt.sampling_methods import LHS
from smt.applications import NestedLHS

import smt.design_space as ds
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

from smt_optim.core.state import State


def generate_initial_design(state: State, evaluator, config) -> None:


    design_space = state.problem.design_space

    # if no initial DoE is defined
    if config.xt_init is None:
        # mono-fidelity problem
        if state.problem.num_fidelity == 1:
            # mixed-variable problem
            if isinstance(design_space, ds.DesignSpace):
                # TODO: issue with seed parameter
                # sampler = MixedIntegerSamplingMethod(LHS,
                #                                      design_space,
                #                                      criterion="ese",
                #                                      seed=config.seed)

                sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(),
                              criterion="ese",
                              seed=config.seed, )


            # continuous problem
            else:
                sampler = LHS(xlimits=design_space,
                              criterion="ese",
                              seed=config.seed)

        # multi-fidelity problem
        # Note: NestedLHS uses criterion="ese"
        else:
            # mixed-variable problem
            if isinstance(design_space, ds.DesignSpace):

                sampler = NestedLHS(design_space=design_space,
                                    nlevel=state.problem.num_fidelity,
                                    seed=config.seed)
            # continuous problem
            else:
                sampler = NestedLHS(xlimits=design_space,
                                    nlevel=state.problem.num_fidelity,
                                    seed=config.seed)

        if config.nt_init is None:
            nt_init = max(5, state.problem.num_dim+1)
        else:
            nt_init = config.nt_init

        doe = sampler(nt_init)

        # quick fix for the seed parameter issue
        if state.problem.num_fidelity == 1 and isinstance(design_space, ds.DesignSpace):
            doe, _ = design_space.fold_x(doe)

        if state.problem.num_fidelity == 1:
            doe = [doe]
            infill = doe * state.problem.num_fidelity
        else:
            infill = doe

    else:
        # TODO: should evaluate validity of initial design space
        infill = copy.deepcopy(config.xt_init)

    evaluator.sample_func(infill, state)

