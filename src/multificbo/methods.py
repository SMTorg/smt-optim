import numpy as np
from typing import Any, Callable, List, Optional, Union

from multificbo.optimizer import Optimizer, ObjectiveConfig, ConstraintConfig, OptimizerConfig
from multificbo.surrogate_models import Surrogate, SmtKRG, SmtMFK, SmtMFCK
from multificbo.acquisition_strategies import MonoFiAcqStrat, MultiFiAcqStrat, VFPI


def sego(objective: Callable, domain: np.ndarray, constraints: list = [],
         max_iter: int = 100,
         xt_init: list = None,
         log: str = "log",
         verbose: bool = False) -> dict:

    # set objective
    obj_config = ObjectiveConfig(
        objective=      objective,
        domain=         domain,
        type=           "minimize",
        surrogate=      SmtKRG,
        costs=          [1],
    )

    # set constraints
    cstr_config = []
    for c_func in constraints:
        # if type(c_func) is not Callable:
        #     print(type(c_func))
        #     raise Exception("Not all constraints are callable.")

        cstr_config.append(
            ConstraintConfig(
                constraint=     c_func,
                type=           "less",
                tol=            1e-4,
                surrogate=      SmtKRG,
            )
        )

    # set optimizer options
    optim_config = OptimizerConfig(
        constraints=    cstr_config,
        max_iter=       max_iter,
        xt_init=        xt_init,
        log_filename=   log,
        verbose=        verbose,
    )

    # set acquisition strategy
    strategy = MonoFiAcqStrat

    # initialize optimizer and start optimization
    optimizer = Optimizer(obj_config, optim_config, strategy)
    opt_data = optimizer.optimize()

    return opt_data


def mfsego(objective: List[Callable], domain: np.ndarray, costs: list, constraints: list = [],
           max_iter: int = 100,
           xt_init: list = None,
           log: str = "log",
           verbose: bool = False) -> dict:

    # set objective
    obj_config = ObjectiveConfig(
        objective=      objective,
        domain=         domain,
        type=           "minimize",
        surrogate=      SmtMFK,
        costs=          costs,
    )

    # set constraints
    cstr_config = []
    for c_funcs in constraints:
        cstr_config.append(
            ConstraintConfig(
                constraint=     c_funcs,
                type=           "less",
                tol=            1e-4,
                surrogate=      SmtMFK,
            )
        )

    # set optimizer options
    optim_config = OptimizerConfig(
        constraints=    cstr_config,
        max_iter=       max_iter,
        xt_init=        xt_init,
        log_filename=   log,
        verbose=        verbose,
    )

    # set acquisition strategy
    strategy = MultiFiAcqStrat

    # initialize optimizer and start optimization
    optimizer = Optimizer(obj_config, optim_config, strategy)
    opt_data = optimizer.optimize()

    return opt_data


def vfpi(objective: List[Callable], domain: np.ndarray, costs: list, constraints: list = [],
         max_iter: int = 100,
         xt_init: list = None,
         log: str = "log",
         verbose: bool = False) -> dict:

    # set objective
    obj_config = ObjectiveConfig(
        objective=      objective,
        domain=         domain,
        type=           "minimize",
        surrogate=      SmtMFCK,
        costs=          costs,
    )

    # set constraints
    cstr_config = []
    for c_funcs in constraints:
        cstr_config.append(
            ConstraintConfig(
                constraint=     c_funcs,
                type=           "less",
                tol=            1e-4,
                surrogate=      SmtMFCK,
            )
        )

    # set optimizer options
    optim_config = OptimizerConfig(
        constraints=    cstr_config,
        max_iter=       max_iter,
        xt_init=        xt_init,
        log_filename=   log,
        verbose=        verbose,
    )

    # set acquisition strategy
    strategy = VFPI

    # initialize optimizer and start optimization
    optimizer = Optimizer(obj_config, optim_config, strategy)
    optimizer.acq_strategy.sub_optimizer = "COBYLA"   # TODO: fix genetic algorithm
    opt_data = optimizer.optimize()

    return opt_data