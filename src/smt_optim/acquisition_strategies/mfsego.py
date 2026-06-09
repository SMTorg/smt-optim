from time import perf_counter
from typing import Callable

import numpy as np
from scipy import optimize as so, stats as stats
from scipy.spatial.distance import cdist

import smt.design_space as ds

from smt_optim.acquisition_functions import log_ei
from smt_optim.surrogate_models.base import Surrogate
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
                                                             self.fidelity_crit)

        else:
            levels = [(state.problem.num_fidelity - 1) for _ in range(num_points)]

        return levels


def build_scipy_constraints(state: State) -> list[dict]:

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



def corrected_predict_variances_all_levels(x_pred: np.ndarray, model, method: str = "max") -> tuple[np.ndarray, list]:
    """
    Predict the variance at all fidelity levels for given prediction points `x_pred`.

    Parameters
    ----------
    x_pred: np.ndarray of shape (num_points, num_dim)
        Prediction points.
    model: MFK
        SMT's MFK model.
    method: str, optional
        Correction method for ill-conditioned models.

    Returns
    -------
    np.ndarray
        Variances for all points at all fidelity levels.
    list[float]
        The squared correlation coefficient for each level.

    Notes
    -----
    If `method` is set to `None`, no correction is performed. If set to `max`, the maximum variance for each level is
    subtracted  to the variance of `x_pred`. If set to `closest`, for each prediction point and each fidelity level
    the variance of the closest training point is subtracted  to the predicted variance.
    """

    noise2, rho2 = model.predict_variances_all_levels(model.X[-1])

    if method is None:
        noise2_corr = np.zeros((x_pred.shape[0], noise2.shape[1]))
    elif method == "max":
        noise2_corr = np.max(noise2, axis=0).reshape(1, -1).repeat(x_pred.shape[0], axis=0)
    elif method == "closest":
        min_dist_idx = np.argmin(cdist(x_pred, model.X[-1]), axis=1)
        noise2_corr = noise2[min_dist_idx, :]
    else:
        raise Exception(f"`{method}` is not a valid method.")

    s2, rho2 = model.predict_variances_all_levels(x_pred)
    s2_corr = np.clip(s2 - noise2_corr, 0, np.inf)

    return s2_corr, rho2



def compute_sigma2_red(x_pred: np.ndarray, surrogate, method="max") -> np.ndarray:
    r"""
    Compute the variance contribution of each fidelity level viewed from the highest fidelity level.

    For a given surrogate model, the output corresponds to

    .. math::
        \sigma_{\text{red}}^2(\ell, \boldsymbol{x}_{i}) = \sum_{\ell'=1}^{\ell}\sigma^2_{(\delta, \ell')}(\boldsymbol{x}_{i}) \prod_{j=\ell'}^{L-1}{\rho^2_j}.

    where :math:`\ell` corresponds to the fidelity level evaluated and :math:`L` to the total number of fidelity level.

    Parameters
    ----------
    x_pred : np.ndarray of shape (num_points, num_dim)
        Prediction points.
    surrogate : Surrogate
        SMT-Optim surrogate model with a model attribute corresponding to a SMT MFK model.
    method : str, optional
        `corrected_predict_variances_all_levels` correction method.

    Returns
    -------
    np.ndarray of shape (num_points, num_level)
        Variance contribution of each level viewed from the highest fidelity level.

    """

    # np.ndarray(num_points, num_levels), list[np.ndarray(num_points)]
    # s2, rho2 = surrogate.model.predict_variances_all_levels(x_pred)
    s2, rho2 = corrected_predict_variances_all_levels(x_pred, surrogate.model, method=method)
    num_levels = s2.shape[1]

    tot_rho2 = np.ones((x_pred.shape[0], num_levels))
    s2_red = np.empty((x_pred.shape[0], num_levels))

    for k in range(num_levels):
        for l in range(k, num_levels-1):
            tot_rho2[:, k] *= rho2[l][:]

        s2_red[:, k] = s2[:, k] * tot_rho2[:, k]

    # np.array(num_points, num_levels)
    return s2_red


def compute_norm_squared_cost(costs: list[float]) -> np.ndarray:
    r"""
    Compute the normalized total squared cost of each fidelity level.

    The output corresponds to:

    .. math::
        \text{cost}_{\ell} = \left(\sum_{\ell'=1}^{\ell}{c_{\ell'}} \; \bigg/ \; \sum_{\ell'=1}^{L}{c_{\ell'}}\right)^{2}.

    where :math:`\ell` corresponds to the fidelity level evaluated and :math:`L` to the total number of fidelity levels.

    Parameters
    ----------
    costs : list of float
        Evaluation cost of each fidelity level.

    Returns
    -------
    np.ndarray
        Normalized total squared cost of each fidelity level.

    """

    num_levels = len(costs)
    tot_costs2 = np.empty(num_levels)

    for k in range(num_levels):
        tot_costs2[k] = np.sum(costs[0:k+1])**2

    # normalize the aggregate costs squared by its maximum
    tot_costs2 /= np.max(tot_costs2)

    return tot_costs2

def compute_norm_sigma2_red(x_pred: np.ndarray, norm_costs2: list[float], surrogate) -> np.ndarray:
    """
    Normalize the variance reduction of each level by their corresponding normalized total squared costs.

    Parameters
    ----------
    x_pred : np.ndarray of shape (num_points, num_dim)
        Prediction points.
    norm_costs2: list of float
        normalized total squared costs.

    surrogate : Surrogate
        SMT-Optim surrogate model with a model attribute corresponding to a SMT MFK model.

    Returns
    -------
    np.ndarray of shape (num_points, num_level)

    """

    num_levels = len(norm_costs2)

    s2_red = compute_sigma2_red(x_pred, surrogate)
    s2_norm = np.empty_like(s2_red)

    for k in range(num_levels):
        s2_norm[:, k] = s2_red[:, k] / norm_costs2[k]



    # if equal 0 somewhere override...
    # has_zero = np.any(s2_norm == 0, axis=1)
    # indices = s2_norm.shape[1] - 1 - np.argmax((s2_norm == 0)[:, ::-1], axis=1)
    # s2_norm[has_zero, indices[has_zero]] = np.inf

    all_zero = np.all(s2_norm == 0, axis=1)
    s2_norm[all_zero, -1] = np.inf

    return s2_norm


def compute_all_s2_red_norm(x_pred: np.ndarray, costs: list[float], surrogates: list) -> list[np.ndarray]:
    """
    Compute the normalized the variance reduction of all models in the `surrogates` list.

    Parameters
    ----------
    x_pred : np.ndarray of shape (num_points, num_dim)
        Prediction points.
    costs: list of float
        Evaluation cost of each fidelity level.
    surrogates : list of Surrogate
        List of SMT-Optim surrogate models.

    Returns
    -------
    List of np.ndarray of shape (num_points, num_level).

    """

    num_pts = x_pred.shape[0]
    num_levels = len(costs)

    norm_costs2 = compute_norm_squared_cost(costs)

    s2_red_norm = [np.empty((num_pts, num_levels)) for _ in range(len(surrogates))]

    for i, surrogate in enumerate(surrogates):
        s2_red_norm[i] = compute_norm_sigma2_red(x_pred, norm_costs2, surrogate)

    return s2_red_norm


def select_fidelity_level(x_pred: np.ndarray, costs: list[float], all_surrogates: list[Surrogate], criterion: str = "pessimistic") -> tuple[np.ndarray, np.ndarray]:
    """
    Select the highest fidelity level to sample based on the `criterion`.

    Parameters
    ----------
    x_pred : np.ndarray of shape(num_points, num_dim)
        Prediction points.
    costs : list of float
        Evaluation cost of each fidelity level.
    all_surrogates : list of Surrogate
        List of surrogate models.
    criterion : str, optional
        Fidelity criterion. The possible values are : "obj-only", "optimistic", "pessimistic", "average", and "cstr-only".

    Returns
    -------
    np.ndarray of shape (num_points,)
        Highest fidelity level to sample based on the `criterion`.
    np.ndarray
        The criterion value for each fidelity level. The index corresponds to the corresponding fidelity level.

    Notes
    -----
    The `obj-only` criterion only evaluates the variance reduction of the first objective model. The `optimistic`
    criterion selects the overall lowest fidelity level from all the models. The `pessimistic` criterion selects
    the overall highest fidelity level from all the models. The `average` criterion averages the variance reduction
    of all the models on a fidelity level basis, and selects the level with the highest averaged variance reduction.
    The `cstr-only` criterion only works for a single constraint.

    """

    num_pts: int = x_pred.shape[0]
    # level: np.ndarray = np.zeros(num_pts)

    if criterion == "obj-only":
        surrogates = [all_surrogates[0]]
        s2_red_norm = compute_all_s2_red_norm(x_pred, costs, surrogates)
        level = s2_red_norm[0].argmax(axis=1)

    elif criterion == "optimistic":
        s2_red_norm = compute_all_s2_red_norm(x_pred, costs, all_surrogates)

        # TODO: make it compatible with multiple infill points
        level = s2_red_norm[0].argmax(axis=1)

        for i in range(1, len(all_surrogates)):
            level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).min(axis=0)

    elif criterion == "pessimistic":
        s2_red_norm = compute_all_s2_red_norm(x_pred, costs, all_surrogates)

        level = s2_red_norm[0].argmax(axis=1)

        for i in range(1, len(all_surrogates)):
            level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).max(axis=0)

    elif criterion == "average":
        # s2_red of each surrogate is normalized by the cost. Should it be normalized after the sum?
        # -> should be the same
        s2_red_norm = compute_all_s2_red_norm(x_pred, costs, all_surrogates)
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

        s2_red_norm = compute_all_s2_red_norm(x_pred, costs, surrogates)

        level = s2_red_norm[0].argmax(axis=1)

    # np.ndarray(num_pts) -> fidelity level for each infill points
    return level, s2_red_norm