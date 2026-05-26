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


class MultiObj(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()

        self.acq_func_gen = kwargs.get("acq_func", init_mpi)
        self.n_start = kwargs.pop("n_start", 20)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))


    def validate_config(self, state):
        pass


    def get_infill(self, state):

        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_start)

        acq_func: Callable = self.acq_func_gen(state)

        cstr_func: list = build_scipy_constraints(state)

        if self.sp_method is None:
            val = np.empty(multi_x0.shape[0])
            c_val = np.empty((multi_x0.shape[0], state.problem.num_cstr))

            for idx in range(multi_x0.shape[0]):
                val[idx] = acq_func(multi_x0[idx, :].reshape(1, -1))

                for jdx in range(state.problem.num_cstr):
                    c_val[idx, jdx] = -cstr_func[jdx]["fun"](multi_x0[idx, :].reshape(1, -1))

            # constrained problem
            if state.problem.num_cstr > 0:
                rscv = np.sqrt(np.sum(np.maximum(c_val, 0) ** 2, axis=1))
                feas_mask = rscv <= 1e-4    # RSCV tolerance

                # no feasible points
                if len(feas_mask) == 0:
                    idx = np.argmin(rscv)

                # at least on feasible point
                else:
                    idx = np.argmax(np.where(rscv, val, -np.inf))

            # unconstrained problem
            else:
                idx = np.argmax(val)
            next_x = multi_x0[idx, :]

        else:
            def sp_wrapper(x):
                x = x.reshape(1, -1)
                return -acq_func(x)

            res = multistart_minimize(sp_wrapper,
                                      bounds=np.array([[0, 1]] * state.problem.num_dim),
                                      constraints=cstr_func,
                                      n_start=self.n_start,
                                      seed=self.seed,
                                      tol=self.sp_tol,
                                      method=self.sp_method, )

            next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill

def build_scipy_constraints(state: State) -> list[dict]:

    # TODO: merge with similar method in mfsego.py

    scipy_cstr = []

    def append_sp_cstr(func: Callable, type: str) -> None:
        scipy_cstr.append({
            "fun": func,
            "type": type,
        })

    def sp_constraint(x, model):
        x = x.reshape(1, -1)
        mu = model.predict_values(x)
        return mu.item()


    for c_id, c_config in enumerate(state.problem.cstr_configs):

        if c_config.equal is not None:
            func = lambda x, f=sp_constraint, value=state.cstr_equal[c_id], m=state.cstr_models[c_id]: f(x, m) - value
            append_sp_cstr(func, "eq")

        else:
            if c_config.lower is not None:
                func = lambda x, f=sp_constraint, value=state.cstr_lower[c_id], m=state.cstr_models[c_id]: - value + f(x, m)
                append_sp_cstr(func, "ineq")

            if c_config.upper is not None:
                func = lambda x, f=sp_constraint, value=state.cstr_upper[c_id], m=state.cstr_models[c_id]: - f(x, m) + value
                append_sp_cstr(func, "ineq")

    return scipy_cstr

