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


class MFSEGO(AcquisitionStrategy):
    """
    Multi-Fidelity Super Efficient Global Optimization (MF-SEGO) strategy.

    This acquisition strategy can perform Efficient Global Optimization (EGO) (unconstrained optimization),
    SEGO (constrained optimization), and MF-SEGO (multi-fidelity unconstrained or constrained optimization).

    It is compatible with various acquisition functions, including:
    - expected improvement,
    - log expected improvement,
    - probability of improvement, and
    - log probability of improvement.

    The constraints are handled by maximizing the acquisition function with respect to predictions from
    constraint surrogate models, instead of using the Probability-of-Improvement approach.

    In the multi-fidelity setting, the acquisition function is first maximized, followed by fidelity level selection.
    This strategy maintains a nested Design of Experiments (DoE), meaning that for each new fidelity level sampled,
    all lower-fidelity levels are also requested to be sampled.

    MF-SEGO offers different fidelity selection criteria:
    - obj-only,
    - optimistic,
    - pessimistic, and
    - average.

    Parameters
    ----------
    state : State
        Optimization state containing surrogate models, data, and problem definition.

    Other Parameters
    ----------------
    acq_func: callable, optional
        Acquisition function used to rank candidate points (default: log_ei).
    n_start: int, optional
        Number of multistart initializations for the inner optimizer. Default: 20.
    fidelity_crit: {"obj-only", "average", "optimistic", "pessimistic"}, optional
        Strategy used to select fidelity level.
    select_fidelity: bool, optional
        If False, always evaluate all fidelity levels.
    sp_method: str, optional
        Optimization method passed to SciPy (e.g., "SLSQP", "COBYLA"). Default = "SLSQP".
    sp_tol: float, optional
        Tolerance for the SciPy optimizer. Default = sqrt(machine epsilon).

    Notes
    -----
    When optimizing a high-dimensional problem, it is recommended to increase the number of
    starting points (`n_start`). The default setting may be insufficient for problems with higher
    dimensions or many constraints.

    This acquisition strategy is designed to work with SMT's surrogate models. In the multi-fidelity setting,
    SMT's MFK model must be used.
    """


    def __init__(self, state: State, **kwargs):
        """
        Initialize the MFSEGO acquisition strategy.

        Parameters
        ----------
        state : State
            Optimization state.

        **kwargs
            Optional configuration parameters. See class docstring for full list.

        Raises
        ------
        TypeError
            If unexpected keyword arguments are provided.
        """
        super().__init__()

        self.acq_context = state
        self.acq_func = kwargs.pop("acq_func", log_ei)                          # expected_improvement, log_ei
        self.fmin_crit = kwargs.pop("fmin_crit", "min_rscv")                    # broken -> to be removed (min_rscv, fmin, mean_rscv)
        # self.sub_optimizer = kwargs.pop("sub_optimizer", "COBYLA")
        self.n_start = kwargs.pop("n_start", None)                              # optimizer multistart
        self.fidelity_crit = kwargs.pop("fidelity_crit", "obj-only")            # obj-only, average, optimistic, pessimistic
        self.select_fidelity = kwargs.pop("select_fidelity", True)              # if set to False, will always sample (LF+HF)
        self.min_rscv_first = kwargs.pop("min_rscv_first", False)               # broken -> to be fixed!
        self.filter_rscv = kwargs.pop("filter_rscv", False)                     # broken -> to be fixed!
        self.optimize_best = kwargs.pop("optimize_best", False)                 # broken -> to be fixed!
        self.relax_constraints = kwargs.pop("relax_constraints", False)         # broken -> to be fixed!
        self.cr_override = kwargs.pop("cr_override", None)                      # override optimizer Cost Ratio
        self.sp_method = kwargs.pop("sp_method", "SLSQP")                       # SciPy optimizer method
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))        # SciPy optimizer tolerance

        self.seed = kwargs.pop("seed", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if state and self.n_start is None:
            self.n_start = 20 # * state.problem.num_dim

        self.fmin: float | None = None  # current best feasible objective value


    def validate_config(self, acq_context: State) -> None:

        if acq_context.problem.num_obj > 1:
            raise Exception("Multi-objective not implemented.")

        if not isinstance(acq_context.design_space, np.ndarray):
            raise Exception("Design space must be a numpy array.")


        obj_required_methods = [
            "predict_values",
            "predict_variances",
        ]

        for method in obj_required_methods:
            if not callable(getattr(acq_context.obj_models[0], method, None)):
                raise TypeError(f"Objective model requires: '{method}' method.")

        cstr_required_methods = [
            "predict_values",
        ]

        for c_id in range(acq_context.problem.num_cstr):
            for method in cstr_required_methods:
                if not callable(getattr(acq_context.cstr_models[c_id], method, None)):
                    raise TypeError(f"Constraint model requires: '{method}' method.")



    def get_infill(self, acq_context: State) -> list[np.ndarray]:
        """
        Compute the next infill point(s) using the acquisition strategy.

        Parameters
        ----------
        acq_context : State
            Current optimization state, including surrogate models and data.

        Returns
        -------
        list of ndarray
            List of selected infill points. Each entry corresponds to a point
            (and potentially a fidelity level, depending on configuration).

        Notes
        -----
        This method:
        - Optimizes the acquisition function using a multistart strategy
        - Applies the selected fidelity criterion if `select_fidelity=True`
        - Uses SciPy optimizers (controlled via `sp_method`, `sp_tol`)
        """

        if isinstance(self.seed, int) or isinstance(self.seed, float):
            self.seed += 1

        acq_data = dict()

        # gets the current best feasible objective value from the scaled dataset
        best_sample = acq_context.get_best_sample(ctol=0., scaled=True)
        self.fmin = best_sample.obj[0]                                          # mono-objective only
        acq_data["fmin"] = self.fmin

        # scipy objective wrapper
        scipy_obj = self.build_scipy_objective(acq_context)

        # scipy constraint wrapper (for scipy, the feasible domain is g >= 0)
        scipy_cstr = self.build_scipy_constraints(acq_context)

        mix_var = False
        for dv in acq_context.problem.design_space.design_variables:
            if not isinstance(dv, ds.FloatVariable):
                mix_var = True
                break

        # TODO: merge continuous and mixvar multistart optimization
        if not mix_var:
            # generate starting points for the multistart optimization
            gen_t0 = perf_counter()
            # multi_x0 = self.generate_multistart_points(optimizer)
            # TODO: initialize sampler in init class method
            sampler = stats.qmc.LatinHypercube(d=acq_context.problem.num_dim, rng=acq_context.iter)
            multi_x0 = sampler.random(self.n_start)
            gen_t1 = perf_counter()
            acq_data["generate_init_points_time"] = gen_t1 - gen_t0

            res = multistart_minimize(scipy_obj,
                                      bounds=np.array([[0, 1]] * acq_context.problem.num_dim),
                                      multi_x0=multi_x0,
                                      constraints=scipy_cstr,
                                      seed=self.seed,
                                      tol=self.sp_tol,
                                      method=self.sp_method,)
        else:
            res = mixvar_multistart_minimize(scipy_obj,
                                             design_space=acq_context.problem.design_space,
                                             constraints=scipy_cstr,
                                             n_start=self.n_start,
                                             method=self.sp_method,
                                             tol=self.sp_tol,
                                             seed=self.seed)

        # next infill location
        next_x = res.x

        # selects highest fidelity level to sample
        fid_crit_t0 = perf_counter()
        level = self.get_fidelity(next_x.reshape(1, -1), acq_context)[0]

        # keeps the DoE nested -> requests sampling all lower fidelity levels
        infills = []
        for lvl in range(acq_context.problem.num_fidelity):
            if lvl <= level:
                infills.append(next_x.copy().reshape(1, -1))
            else:
                infills.append(None)

        fid_crit_t1 = perf_counter()
        acq_data["fid_crit_time"] = fid_crit_t1 - fid_crit_t0

        acq_context.iter_log["acquisition"] = acq_data

        return infills


    def build_scipy_objective(self, acq_context: State) -> Callable:

        def scipy_acq_func(x):
            x = x.reshape(1, -1)
            mu = acq_context.obj_models[0].predict_values(x).item()
            s2 = acq_context.obj_models[0].predict_variances(x).item()
            return -float(self.acq_func(mu, s2, self.fmin))

        return scipy_acq_func

    def build_scipy_constraints(self, state: State) -> list[dict]:

        scipy_cstr = []

        def append_sp_cstr(func: Callable, type: str) -> None:
            scipy_cstr.append({
                "fun": func,
                "type": type,
            })

        # TODO: re-implement constraint relaxation
        # def sp_constraint(x):
        #     x = x.reshape(1, -1)
        #     mu = mu_func(x).item()
        #
        #     if relax:
        #         s = np.sqrt(s2_func(x).item())
        #         if type == "ineq":
        #             return -(mu - 3 * s)
        #         elif type == "eq":
        #             return -(np.abs(mu) - 3 * s)
        #
        #     return -mu
        #
        # return sp_constraint

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

            all_surrogates = state.obj_models
            for c_surrogate in state.cstr_models:
                all_surrogates.append(c_surrogate)

            if self.cr_override is not None:
                costs = self.cr_override
            else:
                costs = state.problem.costs

            levels, s2_red_norm = self.select_fidelity_level(next_x,
                                                             costs,
                                                             all_surrogates,
                                                             self.fidelity_crit)

        else:
            levels = [(state.problem.num_fidelity - 1) for _ in range(num_points)]

        return levels

        self.acq_log["infill_level"] = level

        if state.problem.num_fidelity > 1 and self.select_fidelity:
            self.acq_log["normalized_s2_reduction"] = s2_red_norm

    # acq_data["multi_idx"] = idx
    # acq_data["multi_x"] = multi_x
    # acq_data["multi_f"] = multi_f
    # acq_data["multi_c"] = multi_c
    # acq_data["acq_success"] = success_rate

    # if optimizer.num_cstr > 0:
    #     acq_data["rscv"] = rscv

    # if self.optimize_best:
    #
    #     if optimizer.num_cstr > 0:
    #         optimized_cstr = np.empty(optimizer.num_cstr)
    #
    #     res = so.minimize(scipy_acq_func,
    #                       x0=x_min,
    #                       bounds=optimizer.domain_scaled,
    #                       constraints=scipy_cstr,
    #                       method="SLSQP",
    #                       tol=1e-15,
    #                       options={"maxiter": 50 * optimizer.num_dim})
    #
    #
    #     optimized_x = np.clip(res.x, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])
    #     acq_data["optimized_x"] = optimized_x
    #
    #     optimized_l2 = np.linalg.norm(optimized_x - x_min)
    #     acq_data["optimized_l2"] = optimized_l2
    #
    #     optimized_acq = scipy_acq_func(optimized_x)
    #     acq_data["optimized_acq"] = optimized_acq
    #
    #     if optimizer.num_cstr > 0:
    #         for c_id in range(len(scipy_cstr)):
    #             optimized_cstr[c_id] = -scipy_cstr[c_id]["fun"](optimized_x)
    #             acq_data["optimized_cstr"] = optimized_cstr
    #
    #     # update x_min
    #     x_min = optimized_x

    # log expected values
    # expected_values = np.empty(acq_context.problem.num_cstr+1)
    # expected_values[0] = acq_context.obj_models[0].predict_values(next_x.reshape(1, -1)).item()

    # if acq_context.scaling:
    #     yt_scaled, yt_mean, yt_std = acq_context._standardize_data(acq_context.yt[-1])
    #     expected_values[0] *= yt_std
    #     expected_values[0] +=  yt_mean
    #
    # for c_id, c_surrogate in enumerate(acq_context.cstr_models):
    #     expected_values[c_id+1] = c_surrogate.predict_values(x_min.reshape(1, -1)).item()
    #     if acq_context.scaling:
    #         yt_scaled, yt_mean, yt_std = acq_context._standardize_data(acq_context.ct[-1][:, c_id])
    #         expected_values[c_id+1] *= yt_std
    #
    # acq_data["expected_values"] = expected_values
    #
    # acq_context.iter_data["acquisition"] = acq_data


    # def generate_multistart_points(self, optimizer) -> np.ndarray:
    #
    #     sampler = stats.qmc.LatinHypercube(d=optimizer.domain_scaled.shape[0])
    #
    #     # LHS filter
    #     large_x0 = sampler.random(10*self.n_start)
    #     # large_x0 = sampler.random(self.n_start)
    #     large_x0 = stats.qmc.scale(large_x0, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])
    #
    #     mu = optimizer.obj_models.predict_values(large_x0)
    #     s2 = optimizer.obj_models.predict_variances(large_x0)
    #     large_f = -self.acq_func(mu, s2, self.fmin)
    #
    #     # no constraints -> selects the best starting points
    #     if optimizer.num_cstr == 0:
    #         sorted_idx = np.argsort(large_f.ravel())
    #
    #     # with constraints -> selects the best points with the lowest constraint violation
    #     else:
    #         large_c = np.empty((large_x0.shape[0], optimizer.num_cstr))
    #
    #         for c_id, c_surrogate in enumerate(optimizer.cstr_models):
    #             large_c[:, c_id] = c_surrogate.predict_values(large_x0).ravel()
    #
    #         rscv = self.compute_rscv(large_c, optimizer.cstr_config)
    #         sorted_idx = np.lexsort((large_f.ravel(), rscv))
    #         rscv = rscv[sorted_idx][:self.n_start]
    #
    #     multi_x0 = large_x0[sorted_idx][:self.n_start, :]
    #
    #     # with constraints -> try to reduce the starting point RSCV
    #     if optimizer.num_cstr > 0 and self.min_rscv_first:
    #
    #         def min_rscv(x):
    #             cstr_values = np.empty((1, len(optimizer.cstr_models)))
    #             for c_id, c_surrogate in enumerate(optimizer.cstr_models):
    #                 cstr_values[0, c_id] = c_surrogate.predict_values(x.reshape(1, -1)).item()
    #             return self.compute_rscv(cstr_values, optimizer.cstr_config).item()
    #
    #
    #         for i in range(multi_x0.shape[0]):
    #             # try to reduce the constraint violation if the starting point is not feasible
    #             if rscv[i] != 0.0:
    #                 res = so.minimize(min_rscv,
    #                                   x0=multi_x0[i, :],
    #                                   bounds=optimizer.domain_scaled,
    #                                   method="COBYLA",
    #                                   tol=1e-8,
    #                                   options={"maxiter": 50*optimizer.num_dim})
    #
    #                 rscv[i] = res.fun
    #                 multi_x0[i, :] = res.x
    #
    #     return multi_x0

    def compute_sigma2_red(self, x_pred: np.ndarray, surrogate) -> np.ndarray:

        # np.ndarray(num_points, num_levels), list[np.ndarray(num_points)]
        s2, rho2 = surrogate.model.predict_variances_all_levels(x_pred)
        num_levels = s2.shape[1]

        tot_rho2 = np.ones((x_pred.shape[0], num_levels))
        s2_red = np.empty((x_pred.shape[0], num_levels))

        for k in range(num_levels):
            for l in range(k, num_levels-1):
                tot_rho2[:, k] *= rho2[l][:]

            s2_red[:, k] = s2[:, k] * tot_rho2[:, k]

        # np.array(num_points, num_levels)
        return s2_red


    def compute_norm_squared_cost(self, costs: list[float]) -> np.ndarray:

        num_levels = len(costs)
        tot_costs2 = np.empty(num_levels)

        for k in range(num_levels):
            tot_costs2[k] = np.sum(costs[0:k+1])**2

        # normalize the aggregate costs squared by its maximum
        tot_costs2 /= np.max(tot_costs2)

        return tot_costs2

    def compute_norm_sigma2_red(self, x_pred: np.ndarray, norm_costs2: list[float], surrogate) -> np.ndarray:

        num_levels = len(norm_costs2)

        s2_red = self.compute_sigma2_red(x_pred, surrogate)
        s2_norm = np.empty_like(s2_red)

        for k in range(num_levels):
            s2_norm[:, k] = s2_red[:, k] / norm_costs2[k]

        return s2_norm


    def compute_all_s2_red_norm(self, x_pred: np.ndarray, costs: list[float], surrogates: list) -> list[np.ndarray]:

        num_pts = x_pred.shape[0]
        num_levels = len(costs)

        norm_costs2 = self.compute_norm_squared_cost(costs)

        s2_red_norm = [np.empty((num_pts, num_levels)) for _ in range(len(surrogates))]

        for i, surrogate in enumerate(surrogates):
            s2_red_norm[i] = self.compute_norm_sigma2_red(x_pred, norm_costs2, surrogate)

        return s2_red_norm


    def select_fidelity_level(self, x_pred: np.ndarray, costs: list[float], all_surrogates: list, criterion: str) -> np.ndarray:

        num_pts: int = x_pred.shape[0]
        # level: np.ndarray = np.zeros(num_pts)

        if criterion == "obj-only":
            surrogates = [all_surrogates[0]]
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, surrogates)
            level = s2_red_norm[0].argmax(axis=1)

        elif criterion == "optimistic":
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)

            # TODO: make it compatible with multiple infill points
            level = s2_red_norm[0].argmax(axis=1)

            for i in range(1, len(all_surrogates)):
                level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).min(axis=0)

        elif criterion == "pessimistic":
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)

            level = s2_red_norm[0].argmax(axis=1)

            for i in range(1, len(all_surrogates)):
                level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).max(axis=0)

        elif criterion == "average":
            # s2_red of each surrogate is normalized by the cost. Should it be normalized after the sum?
            # -> should be the same
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)
            s2_red_avg = np.zeros((num_pts, s2_red_norm[0].shape[1]))

            # sum the s2_red from all surrogates
            for i in range(len(all_surrogates)):
                s2_red_avg[:, :] += s2_red_norm[i][:, :]

            level = s2_red_avg.argmax(axis=1)

        elif criterion == "cstr-only":

            if len(all_surrogates) == 1:
                raise Exception("cstr-only criterion requires one constraint surrogate.")

            surrogates = all_surrogates[1:]

            if len(surrogates) > 1:
                raise Exception("cstr-only is not implemented for more than 1 constraints.")

            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, surrogates)

            level = s2_red_norm[0].argmax(axis=1)

        # np.ndarray(num_pts) -> fidelity level for each infill points
        return level, s2_red_norm
