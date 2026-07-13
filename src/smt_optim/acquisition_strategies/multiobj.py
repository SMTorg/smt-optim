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

from smt_optim.acquisition_functions.multi_obj import init_bi_obj_pi


class MultiObj(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()


        self.acq_func_gen = kwargs.get("acq_func", init_bi_obj_pi)
        self.n_start = kwargs.pop("n_start", 20)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))


    def validate_config(self, state):
        pass


    def get_infill(self, state):

        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_start)

        bi_obj_func = self.acq_func_gen(state)

        if self.sp_method is None:
            val = np.empty(multi_x0.shape[0])
            for idx in range(multi_x0.shape[0]):
                val[idx] = bi_obj_func(multi_x0[idx, :].reshape(1, -1))
            idx = np.argmax(val)
            next_x = multi_x0[idx, :]

        else:
            def sp_wrapper(x):
                x = x.reshape(1, -1)
                return -bi_obj_func(x)

            res = multistart_minimize(sp_wrapper,
                                      bounds=np.array([[0, 1]] * state.problem.num_dim),
                                      constraints=[],
                                      n_start=self.n_start,
                                      seed=self.seed,
                                      tol=self.sp_tol,
                                      method=self.sp_method, )

            next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill

