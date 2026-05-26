from time import perf_counter
from typing import Callable

import numpy as np
from scipy import stats as stats

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import smt.design_space as ds

from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core.state import State
from smt_optim.subsolvers.multistart import mixvar_multistart_minimize

from smt_optim.subsolvers import multistart_minimize

from smt_optim.acquisition_functions import init_ehvi_2o

from smt_optim.acquisition_strategies.mfsego import (
    build_scipy_constraints,
    select_fidelity_level,
)

from smt_optim.utils.multi_obj import PymooStateWrapper


class MOSEGO(AcquisitionStrategy):
    """
    Multi-objective Super Efficient Global Optimization acquisition strategy.

    This acquisition strategy can perform multi-objective optimization on unconstrained, constrained, and
    multi-fidelity optimization problems.

    The constraints are handled by maximizing the acquisition function with respect to predictions from
    constraint surrogate models, instead of using the Probability-of-Improvement approach.

    In the multi-fidelity setting, the acquisition function is first maximized, followed by fidelity level selection.
    This strategy maintains a nested Design of Experiments (DoE), meaning that for each new fidelity level sampled,
    all lower-fidelity levels are also requested to be sampled.

        Parameters
    ----------
    state : State
        Optimization state containing the current problem definition,
        surrogate models, and dataset.

    acq_init : Callable, optional
        Acquisition function initializer used to generate the acquisition
        function. By default, uses `init_ehvi_2o`.

    n_start : int, optional
        Number of starting points used for local acquisition optimization.
        Default is 20.

    genetic_flag : bool, optional
        Whether to use a genetic algorithm for global acquisition optimization.
        Default is True.

    genetic_pop_size : int, optional
        Population size for the genetic optimizer. Default is 20.

    genetic_n_gen : int, optional
        Number of generations for the genetic optimizer. Default is 100.

    sp_method : str, optional
        Local optimization method used for acquisition refinement.
        Default is "SLSQP".

    sp_tol : float, optional
        Tolerance for the local optimizer. Default is
        `sqrt(np.finfo(float).eps)`.

    select_fidelity : bool, optional
        Whether to optimize fidelity selection when using multi-fidelity
        optimization. Default is True.

    cr_override : float or None, optional
        Optional override value for the cost ratio used in fidelity selection.
        Default is None.

    seed : int or None, optional
        Random seed for reproducibility. Default is None.


    Notes
    -----
    All objective configuration type must be set to `minimize`.

    If `genetic_flag` is set to `True`, `genetic_pop_size` points are added to the `n_start` points generated with LHS.
    These points are obtained by solving the predicted Pareto Front (PF) using the surrogate models. `genetic_flag` is
    currently not available for mixed-variable problems.
    """

    def __init__(
        self,
        state: State,
        acq_init: Callable = init_ehvi_2o,
        n_start: int = 20,
        genetic_flag: bool = True,
        genetic_pop_size: int = 20,
        genetic_n_gen: int = 100,
        sp_method: str = "SLSQP",
        sp_tol: float = np.sqrt(np.finfo(float).eps),
        select_fidelity: bool = True,
        cr_override: float | None = None,
        var_red_corr: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.acq_init = acq_init
        self.n_start = n_start
        self.genetic_flag = genetic_flag
        self.genetic_pop_size = genetic_pop_size
        self.genetic_n_gen = genetic_n_gen
        self.sp_method = sp_method
        self.sp_tol = sp_tol
        self.select_fidelity = select_fidelity
        self.cr_override = cr_override
        self.var_red_corr = var_red_corr
        self.seed = seed

        # TODO: work required to adapt console logging and PymooStateWrapper
        for i, obj_config in enumerate(state.problem.obj_configs):
            if obj_config.type != "minimize":
                raise ValueError(
                    f"MOSEGO currently supports minimization objectives only, but objective "
                    f"{i} has type '{obj_config.type}'."
                )

    def validate_config(self, state):
        pass

    def get_infill(self, state):

        if isinstance(self.seed, int) or isinstance(self.seed, float):
            self.seed += 1

        acq_data = dict()

        acq_func: Callable = self.acq_init(state)

        def scipy_obj(x):
            x = x.reshape(1, -1)
            return -acq_func(x)

        scipy_cstr: list = build_scipy_constraints(state)

        mix_var = False
        for dv in state.problem.design_space.design_variables:
            if not isinstance(dv, ds.FloatVariable):
                mix_var = True
                break

        # TODO: merge continuous and mixvar multistart optimization
        if not mix_var:
            # generate starting points for the multistart optimization
            gen_t0 = perf_counter()
            # TODO: initialize sampler in init class method
            sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, seed=state.iter)

            multi_x0 = sampler.random(self.n_start)

            if self.genetic_flag:
                pymoo_prob = PymooStateWrapper(state, scaled=True, train=False)
                algorithm = NSGA2(pop_size=self.genetic_pop_size, seed=self.seed)
                res = minimize(
                    pymoo_prob, algorithm, ("n_gen", self.genetic_n_gen), seed=self.seed
                )

                multi_x0 = np.vstack(
                    (
                        multi_x0,
                        res.X,
                    )
                )

            gen_t1 = perf_counter()
            acq_data["generate_init_points_time"] = gen_t1 - gen_t0

            res = multistart_minimize(
                scipy_obj,
                bounds=np.array([[0, 1]] * state.problem.num_dim),
                multi_x0=multi_x0,
                constraints=scipy_cstr,
                seed=self.seed,
                tol=self.sp_tol,
                method=self.sp_method,
            )

        else:
            res = mixvar_multistart_minimize(
                scipy_obj,
                design_space=state.problem.design_space,
                constraints=scipy_cstr,
                n_start=self.n_start,
                method=self.sp_method,
                tol=self.sp_tol,
                seed=self.seed,
            )

        next_x = res.x

        # selects highest fidelity level to sample
        fid_crit_t0 = perf_counter()
        level = self.get_fidelity(next_x.reshape(1, -1), state)[0]

        # keeps the DoE nested -> requests sampling all lower fidelity levels
        infills = []
        for lvl in range(state.problem.num_fidelity):
            if lvl <= level:
                infills.append(next_x.copy().reshape(1, -1))
            else:
                infills.append(None)

        fid_crit_t1 = perf_counter()
        acq_data["fid_crit_time"] = fid_crit_t1 - fid_crit_t0

        state.iter_log["acquisition"] = acq_data

        return infills

    def get_fidelity(self, next_x: np.ndarray, state: State) -> list[int]:
        """
        Select the highest fidelity level to sample at the given point(s).

        Parameters
        ----------
        next_x : np.ndarray
            The point(s) to sample at.
        state : State
            The current optimization state.

        Returns
        -------
        levels : list[int] or array of int
            The selected fidelity level(s). If `state.problem.num_fidelity` is 1, returns the single fidelity level;
            otherwise, returns a list of fidelity levels, one for each point in `next_x`.

        Notes
        -----
        This method takes into account the problem's cost model and the available surrogate models.
        """

        num_points = next_x.shape[0]

        if state.problem.num_fidelity > 1 and self.select_fidelity:
            all_surrogates = []
            for o_surrogate in state.obj_models:
                all_surrogates.append(o_surrogate)
            for c_surrogate in state.cstr_models:
                all_surrogates.append(c_surrogate)

            if self.cr_override is not None:
                costs = self.cr_override
            else:
                costs = state.problem.costs

            levels, s2_red_norm = select_fidelity_level(
                next_x,
                costs,
                all_surrogates,
                "pessimistic",
                self.var_red_corr,
            )

        else:
            levels = [(state.problem.num_fidelity - 1) for _ in range(num_points)]

        return levels
