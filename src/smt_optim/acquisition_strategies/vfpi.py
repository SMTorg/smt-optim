from time import perf_counter
from typing import Callable

import numpy as np
from scipy import optimize as so, stats as stats

from smt_optim.acquisition_functions import probability_of_improvement, fidelity_correlation
from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core.state import State

from smt_optim.utils.get_fmin import get_fmin

from smt_optim.subsolvers import multistart_minimize


class VFPI(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()
        self.n_start = kwargs.pop("n_start", None)
        # self.cr_override = kwargs.pop("cr_override", None)                  # override optimizer Cost Ratio

        self.seed = kwargs.pop("seed", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if state and self.n_start is None:
            self.n_start = 10 * state.problem.num_dim

    def validate_config(self, acq_context: State) -> None:
        pass


    def get_infill(self, state: State) -> list[np.ndarray]:

        acq_data = dict()

        # get predicted f_min
        self.f_min = self.get_predicted_fmin(state)


        # get infill_x and infill_fidelity
        best_epi_f = -np.inf
        best_epi_x = None
        best_epi_lvl = None

        for lvl in range(state.problem.num_fidelity):
            # setup EPI
            func = lambda x, l=lvl, s=state: -self.epi(x, l, s)
            res = multistart_minimize(func, state.problem.design_space)

            if -res.fun > best_epi_f:
                best_epi_f = -res.fun
                best_epi_x = res.x
                best_epi_lvl = lvl

        infills = []
        for lvl in range(state.problem.num_fidelity):
            if lvl == best_epi_lvl:
                infills.append(best_epi_x.copy().reshape(1, -1))
            else:
                infills.append(None)

        return infills



    def get_predicted_fmin(self, state):

        def obj_wrapper(x):
            y = state.obj_models[0].predict_values(x.reshape(1, -1))
            return y.item()

        res = multistart_minimize(obj_wrapper,
                                  state.problem.design_space,
                                  n_start=self.n_start,
                                  seed=self.seed)

        return res.fun

    def epi(self, x: np.ndarray, lvl: int, state: State) -> float:

        x = x.reshape(1, -1)

        mu, s2 = state.obj_models[0].model.predict_all_levels(x)
        cov = state.obj_models[0].predict_level_covariances(x, lvl)

        # probability of improvement
        pi = probability_of_improvement(mu[-1].reshape(-1, 1), s2[-1].reshape(-1, 1), self.f_min)

        # fidelity correlation penalty
        corr = fidelity_correlation(cov, s2[lvl].reshape(-1, 1), s2[-1].reshape(-1, 1))

        # cost ratio penalty
        cost_ratio = state.problem.costs[-1]/state.problem.costs[lvl]

        # density penalty
        density = self.sample_density(x, lvl, state.obj_models[0].model)

        # probability of feasibility
        pof = 1.0

        for c_id in range(state.problem.num_cstr):
            # g_pred = self.optimizer.cstr_surrogates[c_id].predict_values(x)
            # s2_pred = self.optimizer.cstr_surrogates[c_id].predict_variances(x)

            # TODO: add predict_all_levels() to mfck wrapper
            g_pred, s2_pred = state.cstr_models[c_id].mfck.predict_all_levels(x)

            pof *= stats.norm.cdf(-g_pred[lvl] / np.sqrt(s2_pred[lvl].reshape(1, 1)))

        return (pi * corr * cost_ratio * density * pof).item()


    def sample_density(self, x: np.ndarray, lvl: int, mfck):

        x_scale = mfck.X_scale
        x_offset = mfck.X_offset

        x = (x - x_offset) / x_scale

        xt_lvl = mfck.X[lvl]
        xt_lvl = (xt_lvl - x_offset) / x_scale
        dim = xt_lvl.shape[1]

        optimal_theta = mfck.optimal_theta

        if lvl == 0:
            sigma2 = optimal_theta[0]
            theta = optimal_theta[1:dim + 1]
        else:
            start = (dim + 1) + (2 + dim) * (lvl - 1)
            end = (dim + 1) + (2 + dim) * (lvl)
            sigma2 = optimal_theta[start]
            theta = optimal_theta[start + 1:end - 1]

        R = 1 - mfck._compute_K(x, xt_lvl, (sigma2, theta)) / sigma2
        penalty = np.prod(R, axis=1).reshape(-1, 1)

        return penalty

