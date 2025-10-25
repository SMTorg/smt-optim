import numpy as np
from typing import Any, Callable, List, Optional, Union

from multificbo.optimizer import Optimizer, ObjectiveConfig, ConstraintConfig, OptimizerConfig
from multificbo.surrogate_models import Surrogate, SmtKRG, SmtMFK, SmtMFCK
from multificbo.acquisition_strategies import MonoFiAcqStrat, MultiFiAcqStrat, MFEI, VFPI


def sego(objective: Callable, domain: np.ndarray, constraints: list = [],
         max_iter: int = 100,
         max_budget: float = np.inf,
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
        max_budget=     max_budget,
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
           max_budget: float = np.inf,
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
        max_budget=     max_budget,
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
         max_budget: float = np.inf,
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
        max_budget=     max_budget,
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

def run_optimizer(
        objective,
        domain,
        surrogate,
        strategy,
        costs,
        constraints: list = [],
        max_iter: int = np.inf,
        max_budget: float = np.inf,
        xt_init: list = None,
        log: str = "log",
        verbose: bool = False,
        optimizer_kwargs: dict = None,
        strategy_kwargs: dict = None
) -> dict:

    if max_iter == np.inf and max_budget == np.inf:
        raise Exception("At least one stopping criterion must be defined.")

    obj_config = ObjectiveConfig(
        objective=objective,
        domain=domain,
        type="minimize",
        surrogate=surrogate,
        costs=costs,
    )

    cstr_config = [
        ConstraintConfig(
            constraint=c_funcs,
            type="less",
            tol=1e-4,
            surrogate=surrogate,
        )
        for c_funcs in constraints
    ]

    optim_config = OptimizerConfig(
        constraints=cstr_config,
        max_iter=max_iter,
        max_budget=max_budget,
        xt_init=xt_init,
        log_filename=log,
        verbose=verbose,
    )

    optimizer = Optimizer(obj_config, optim_config, strategy)

    if optimizer_kwargs:
        for key, value in optimizer_kwargs.items():
            setattr(optimizer, key, value)

    return optimizer.optimize()


def minimize(
        objective,
        domain,
        costs,
        method: str,
        max_iter: int = np.inf,
        max_budget: float = np.inf,
        constraints: list = [],
        xt_init: list = None,
        log: str = "log",
        verbose: bool = False,
        optimizer_kwargs: dict = None,
        strategy_kwargs: dict = None
):

    methods = {
        "sego": dict(surrogate=SmtKRG, strategy=MonoFiAcqStrat, costs=[1]),
        "mfsego": dict(surrogate=SmtMFK, strategy=MultiFiAcqStrat),
        "vfpi": dict(surrogate=SmtMFCK, strategy=VFPI),
        "mfei": dict(surrogate=SmtMFCK, strategy=MFEI),
    }

    config = methods[method]
    surrogate = config["surrogate"]
    strategy = config["strategy"]
    costs = costs or config["costs", [1]]

    return run_optimizer(
        objective=objective,
        domain=domain,
        surrogate=surrogate,
        strategy=strategy,
        costs=costs,
        constraints=constraints,
        max_iter=max_iter,
        max_budget=max_budget,
        xt_init=xt_init,
        log=log,
        verbose=verbose,
        optimizer_kwargs=optimizer_kwargs,
        strategy_kwargs=strategy_kwargs,
    )
