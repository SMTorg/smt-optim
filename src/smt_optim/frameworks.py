import numpy as np
from typing import Any, Callable, List, Optional, Union

import smt.design_space as ds

from smt_optim.core import Driver, ObjectiveConfig, ConstraintConfig, DriverConfig, Problem, State
from smt_optim.surrogate_models import SmtAutoModel,  SmtMFCK
from smt_optim.acquisition_strategies import MFSEGO, VFPI


def minimize(
        objective: list[Callable],
        design_space: ds.DesignSpace | np.ndarray,
        method: str,
        costs: list = [1],
        max_iter: int = 100,
        max_budget: int = np.inf,
        constraints: list = [],
        driver_kwargs: dict = {},
        strategy_kwargs: dict = {},
        verbose: bool = True,
) -> State:

    methods = {
        "ego": dict(surrogate=SmtAutoModel, strategy=MFSEGO, costs=[1]),
        "sego": dict(surrogate=SmtAutoModel, strategy=MFSEGO, costs=[1]),
        "mfsego": dict(surrogate=SmtAutoModel, strategy=MFSEGO),
        "vfpi": dict(surrogate=SmtMFCK, strategy=VFPI),
    }

    config = methods[method]
    surrogate = config["surrogate"]
    strategy = config["strategy"]
    costs = costs or config["costs", [1]]

    # ------- setup objective configuration -------
    obj_config = ObjectiveConfig(
        objective,
        type="minimize",
        surrogate=surrogate,
    )

    # ------- setup constraint configurations -------
    cstr_configs = []
    for c_dict in constraints:
        cstr_configs.append(
            ConstraintConfig(
                c_dict["fun"],
                equal = c_dict["equal"] if c_dict.get("equal", None) is not None else None,
                lower = c_dict["lower"] if c_dict.get("lower", None) is not None else None,
                upper = c_dict["upper"] if c_dict.get("upper", None) is not None else None,
                surrogate=surrogate,
            )
        )

    # ------- problem configuration -------
    problem = Problem(
        obj_configs=[obj_config],
        design_space=design_space,
        costs=costs,  # Set the cost of sampling each level
        cstr_configs=cstr_configs,
    )

    # ------- driver configuration -------
    default_kwargs = {
        "max_iter": max_iter,
        "max_budget": max_budget,
        "verbose": verbose,
        "scaling": True,
    }

    # overrides defaults if key collide
    driver_kwargs = {**default_kwargs, **driver_kwargs}

    driver_config = DriverConfig(
        **driver_kwargs,
    )

    # ------- start driver -------
    driver = Driver(problem, driver_config, strategy, strategy_kwargs=strategy_kwargs)
    state = driver.optimize()
    return state