import numpy as np
from typing import Callable

import smt.design_space as ds

from smt_optim.core import (
    Driver,
    ObjectiveConfig,
    ConstraintConfig,
    DriverConfig,
    Problem,
    State,
)
from smt_optim.surrogate_models.smt import SmtAutoModel, SmtGPX, SmtMFCK
from smt_optim.acquisition_strategies import MFSEGO, VFPI
from smt_optim.acquisition_strategies.biego import BiEGO


def minimize(
    objective: list[Callable] | list[list[Callable]],
    design_space: ds.DesignSpace | np.ndarray,
    method: str | None = None,
    costs: list | None = None,
    max_iter: int = 100,
    max_budget: float = np.inf,
    constraints: list | None = None,
    driver_kwargs: dict | None = None,
    strategy_kwargs: dict | None = None,
    verbose: bool = True,
) -> State:
    """
    Minimize the objective function with respect to the problem properties.

    This function provides a unified interface to perform optimization using
    different acquisition-based strategies (e.g., EGO, SEGO, MFSEGO, VFPI).
    It supports mono-fidelity and multi-fidelity optimization, with optional
    constraints and budget control.

    Parameters
    ----------
    objective: list[Callable] | list[list[Callable]]
        List of objective function callables ordered by increasing fidelity. For multi-objective
        optimization, provide a list of lists (one list of fidelity functions per objective).
    design_space: ds.DesignSpace | np.ndarray
        Problem design space. If a np.ndarray is provided, the problem will be treated as fully continuous.
    method: str {"ego", "sego", "mfsego", "vfpi"} or None, optional
        Optimization framework to use. If None, the method is selected automatically based on the problem
        characteristics  between {"ego", "sego", "mfsego"}.
    costs: list[float], optional
        Evaluation cost associated with each fidelity level, ordered from
        lowest to highest fidelity. Required for multi-fidelity optimization.
        Defaults to [1] for mono-fidelity problems.
    max_iter: int, default=100
        Maximum number of optimization iteration.
    max_budget: float, default=np.inf
        Maximum total evaluation budget.The optimization stops when this budget is exhausted.
    constraints: list[dict], optional
        List of the constraint definitions.
    driver_kwargs: dict, optional
        Additional keyword arguments passed to the optimization driver.
    strategy_kwargs: dict, optional
        Additional keyword arguments passed to the acquisition strategy.
    verbose: bool
        If True, prints progress information during optimization.

    Returns
    -------
    State
        Final optimization state.
    """

    if method is None:
        strategy = MFSEGO

        if isinstance(design_space, ds.DesignSpace):
            mix_var = True
        else:
            mix_var = False

        # Check if objective is a list of lists (Multi-Objective)
        if len(objective) > 0 and isinstance(objective[0], list):
            is_multi_obj = True
            if len(objective[0]) > 1:
                multi_fidelity = True
            else:
                multi_fidelity = False
        else:
            is_multi_obj = False
            if len(objective) > 1:
                multi_fidelity = True
            else:
                multi_fidelity = False

        if not mix_var and not multi_fidelity:
            surrogate = SmtGPX
        else:
            surrogate = SmtAutoModel

        if multi_fidelity:
            expected_costs_len = len(objective[0]) if is_multi_obj else len(objective)
            if len(costs) != expected_costs_len:
<<<<<<< HEAD
                raise Exception(
                    "Error: len(costs) does not match the number of fidelity levels in objective"
                )
=======
                raise Exception("Error: len(costs) does not match the number of fidelity levels in objective")
>>>>>>> upstream/main

    else:
        methods = {
            "ego": dict(surrogate=SmtAutoModel, strategy=MFSEGO, costs=[1]),
            "sego": dict(surrogate=SmtAutoModel, strategy=MFSEGO, costs=[1]),
            "mfsego": dict(surrogate=SmtAutoModel, strategy=MFSEGO),
            "vfpi": dict(surrogate=SmtMFCK, strategy=VFPI),
        }

        config = methods[method]
        surrogate = config["surrogate"]
        strategy = config["strategy"]
        costs = costs or config.get("costs", [1])

        if constraints is None:
            constraints = []
        if driver_kwargs is None:
            driver_kwargs = {}
        if strategy_kwargs is None:
            strategy_kwargs = {}

    # ------- setup objective configuration -------
    obj_configs = []
    # If it's a list of lists, it's multi-objective
    if len(objective) > 0 and isinstance(objective[0], list):
        for obj_list in objective:
            obj_configs.append(
                ObjectiveConfig(
                    obj_list,
                    type="minimize",
                    surrogate=surrogate,
                )
            )
    else:
        obj_configs.append(
            ObjectiveConfig(
                objective,
                type="minimize",
                surrogate=surrogate,
            )
        )

    # ------- setup constraint configurations -------
    cstr_configs = []
    for c_dict in constraints:
        cstr_configs.append(
            ConstraintConfig(
                c_dict["fun"],
                equal=c_dict["equal"]
                if c_dict.get("equal", None) is not None
                else None,
                lower=c_dict["lower"]
                if c_dict.get("lower", None) is not None
                else None,
                upper=c_dict["upper"]
                if c_dict.get("upper", None) is not None
                else None,
                surrogate=surrogate,
            )
        )

    # ------- problem configuration -------
    problem = Problem(
        obj_configs=obj_configs,
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
    if strategy_kwargs is None:
        strategy_kwargs = {}

    driver = Driver(problem, driver_config, strategy, strategy_kwargs=strategy_kwargs)
    state = driver.optimize()
    return state
