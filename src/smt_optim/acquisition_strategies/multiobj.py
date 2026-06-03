from time import perf_counter
from typing import Callable

import numpy as np
from scipy import optimize as so, stats as stats

import smt.design_space as ds

from smt_optim.acquisition_functions import log_ei
from smt_optim.acquisition_strategies import AcquisitionStrategy
# from smt_optim.surrogate_models.smt import SmtMFK

from smt_optim.core.state import State
from smt_optim.subsolvers.multistart import mixvar_multistart_minimize

from smt_optim.utils.get_fmin import get_fmin

from smt_optim.subsolvers import multistart_minimize

from smt_optim.acquisition_functions.multi_obj import init_mpi

from smt_optim.acquisition_strategies.mfsego import build_scipy_constraints, select_fidelity_level


class MOSEGO(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()

        self.acq_func_gen = kwargs.get("acq_func", init_mpi)
        self.n_start = kwargs.pop("n_start", 20)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))
        self.select_fidelity = kwargs.pop("select_fidelity", True)
        self.cr_override = kwargs.pop("cr_override", None)
        self.seed = kwargs.pop("seed", None)


    def validate_config(self, state):
        pass


    def get_infill(self, state):

        if isinstance(self.seed, int) or isinstance(self.seed, float):
            self.seed += 1

        acq_data = dict()

        acq_func: Callable = self.acq_func_gen(state)

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
            # multi_x0 = self.generate_multistart_points(optimizer)
            # TODO: initialize sampler in init class method
            sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, seed=state.iter)
            multi_x0 = sampler.random(self.n_start)
            gen_t1 = perf_counter()
            acq_data["generate_init_points_time"] = gen_t1 - gen_t0

            res = multistart_minimize(scipy_obj,
                                      bounds=np.array([[0, 1]] * state.problem.num_dim),
                                      multi_x0=multi_x0,
                                      constraints=scipy_cstr,
                                      seed=self.seed,
                                      tol=self.sp_tol,
                                      method=self.sp_method,)
        else:
            res = mixvar_multistart_minimize(scipy_obj,
                                             design_space=state.problem.design_space,
                                             constraints=scipy_cstr,
                                             n_start=self.n_start,
                                             method=self.sp_method,
                                             tol=self.sp_tol,
                                             seed=self.seed)

        next_x = res.x
        print(f"next_x = {next_x}")

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

            levels, s2_red_norm = select_fidelity_level(next_x,
                                                        costs,
                                                        all_surrogates,
                                                        "pessimistic")

            print(f"level(1) = {levels} | {s2_red_norm}")

        else:
            levels = [(state.problem.num_fidelity - 1) for _ in range(num_points)]

        return levels

