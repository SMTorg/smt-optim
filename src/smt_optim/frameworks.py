import numpy as np
from typing import Any, Callable, List, Optional, Union

from smt_optim.optimizer import Optimizer, ObjectiveConfig, ConstraintConfig, OptimizerConfig
from smt_optim.surrogate_models import Surrogate, SmtKRG, SmtMFK, SmtMFCK
from smt_optim.acquisition_strategies import MonoFiAcqStrat, MultiFiAcqStrat, MFSEGO, MFEI, VFPI


def run_optimizer(
        objective,
        domain,
        surrogate,
        strategy,
        costs,
        constraints: list = [],
        max_iter: int = None,
        optimizer_kwargs: dict = None,
        strategy_kwargs: dict = None
) -> tuple[dict, Optimizer]:

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
        verbose=True,
    )

    if optimizer_kwargs:
        for key, value in optimizer_kwargs.items():
            setattr(optim_config, key, value)

    optimizer = Optimizer(obj_config, optim_config, strategy, strategy_kwargs)

    opt_data = optimizer.optimize()

    return opt_data, optimizer


def minimize(
        objective,
        domain,
        costs,
        method: str,
        max_iter: int = np.inf,
        constraints: list = [],
        optimizer_kwargs: dict = None,
        strategy_kwargs: dict = None
):

    methods = {
        "sego": dict(surrogate=SmtKRG, strategy=MFSEGO, costs=[1]),
        "mfsego": dict(surrogate=SmtMFK, strategy=MFSEGO),
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
        optimizer_kwargs=optimizer_kwargs,
        strategy_kwargs=strategy_kwargs,
    )
