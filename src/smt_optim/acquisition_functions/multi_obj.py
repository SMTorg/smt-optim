from functools import partial
from typing import Callable

import numpy as np

import scipy.stats as stats

from smt_optim.utils.multi_obj import get_pf_from_dataset


def init_mpi(state) -> Callable:
    """
    Initialize the Minimum Probability of Improvement (MPoI / MPI) multi-objective acquisition function.
    DOI: https://doi.org/10.1145/3071178.3071276

    Parameters
    ----------
    state: State

    Returns
        Callable acquisition function
    -------

    Notes
    -----
    Uses the scaled dataset
    """

    pareto_front = get_pf_from_dataset(
        state.scaled_dataset
    )  # shape: (n_points, n_objective)

    # if no feasible point in pareto front (possible in constrained optimization)
    if pareto_front.shape[0] == 0:
        data = state.scaled_dataset.export_as_dict()
        min_rscv_idx = np.argmin(data["rscv"])
        pareto_front = data["obj"][min_rscv_idx, :].reshape(1, -1)

    def mpi_func(x: np.ndarray) -> float:
        """
        Minimum Probability of Improvement (MPoI / MPI) multi-objective acquisition function
        DOI: https://doi.org/10.1145/3071178.3071276

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        float
            Minimum Probability of Improvement value.
        """

        values = np.ones(pareto_front.shape[0])

        for idx in range(state.problem.num_obj):
            mu_obj = state.obj_models[idx].predict_values(x).item()
            s_obj = state.obj_models[idx].predict_variances(x).item()

            if s_obj > 0:
                s_obj = np.sqrt(s_obj)
            else:
                return 0.0

            values *= stats.norm.cdf((mu_obj - pareto_front[:, idx]) / s_obj)

        return 1 - np.max(values)

    return mpi_func


def bi_obj_pi(x_pred: np.ndarray, obj_vals: np.ndarray, models: list) -> np.ndarray:
    """
    Compute the bi-objective Probability of Improvement (PI).

    Parameters
    ----------
    x_pred : np.ndarray
        The input features for prediction, shape (n_samples, n_features).
    obj_vals : np.ndarray
        The Pareto front objective values, shape (n_pareto, 2).
    models : list
        List of two surrogate models for the two objectives.

    Returns
    -------
    np.ndarray
        The computed Probability of Improvement values, shape (n_samples, 1).
    """
    s0_sq = models[0].predict_variances(x_pred)
    s1_sq = models[1].predict_variances(x_pred)

    s0 = np.sqrt(np.maximum(s0_sq, 0.0))
    s1 = np.sqrt(np.maximum(s1_sq, 0.0))

    y0 = models[0].predict_values(x_pred)
    y1 = models[1].predict_values(x_pred)

    pi = np.zeros((x_pred.shape[0], 1))

    valid_mask = ((s0 > 0.0) & (s1 > 0.0)).ravel()
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return pi

    s0_v = s0[valid_idx]
    s1_v = s1[valid_idx]
    y0_v = y0[valid_idx]
    y1_v = y1[valid_idx]

    z0 = (obj_vals[:, 0].reshape(1, -1) - y0_v) / s0_v
    z1 = (obj_vals[:, 1].reshape(1, -1) - y1_v) / s1_v

    from scipy.special import ndtr

    cdf_0 = ndtr(z0)
    cdf_1 = ndtr(z1)

    pi_v = cdf_0[:, 0].copy()
    if obj_vals.shape[0] > 1:
        diff_cdf_0 = cdf_0[:, 1:] - cdf_0[:, :-1]
        # FIX: use cdf_1[:, :-1] because intervals (f_0^{i-1}, f_0^i] are bounded by f_1^{i-1}
        pi_v += np.sum(diff_cdf_0 * cdf_1[:, :-1], axis=1)

    pi_v += (1.0 - cdf_0[:, -1]) * cdf_1[:, -1]

    pi[valid_idx, 0] = pi_v
    return pi


def init_bi_obj_pi(state, kwargs=None):
    """
    Initialize the bi-objective Probability of Improvement (PI) function.

    Parameters
    ----------
    state : State
        The current optimization state containing dataset and models.
    kwargs : dict, optional
        Additional arguments, by default None.

    Returns
    -------
    Callable
        The initialized bi_obj_pi acquisition function.
    """

    pareto = get_pf_from_dataset(state.scaled_dataset)
    if pareto.shape[0] == 0:
        data = state.scaled_dataset.export_as_dict()
        min_rscv_idx = np.argmin(data["rscv"])
        pareto = data["obj"][min_rscv_idx, :].reshape(1, -1)
    # first objective needs to be sorted? what about the second to compute Y1?
    sorted_idx = np.argsort(pareto[:, 0])
    sorted_pareto = pareto[sorted_idx, :]

    models = state.obj_models

    bi_obj_pi_func = partial(bi_obj_pi, obj_vals=sorted_pareto, models=models)

    return bi_obj_pi_func


def bi_obj_ei(x_pred: np.ndarray, obj_vals: np.ndarray, models: list) -> np.ndarray:
    """
    Compute the bi-objective Expected Improvement (EI).

    Parameters
    ----------
    x_pred : np.ndarray
        The input features for prediction, shape (n_samples, n_features).
    obj_vals : np.ndarray
        The Pareto front objective values, shape (n_pareto, 2).
    models : list
        List of two surrogate models for the two objectives.

    Returns
    -------
    np.ndarray
        The computed Expected Improvement values, shape (n_samples, 1).

    Notes
    -----
    The formulation computes the conditional expectation of improvement over the current
    Pareto front.
    """
    s0_sq = models[0].predict_variances(x_pred)
    s1_sq = models[1].predict_variances(x_pred)

    s0 = np.sqrt(np.maximum(s0_sq, 0.0))
    s1 = np.sqrt(np.maximum(s1_sq, 0.0))

    y0 = models[0].predict_values(x_pred)
    y1 = models[1].predict_values(x_pred)

    EI = np.zeros((x_pred.shape[0], 1))

    valid_mask = ((s0 > 1.0e-15) & (s1 > 1.0e-15)).ravel()
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return EI

    s0_v = s0[valid_idx]
    s1_v = s1[valid_idx]
    y0_v = y0[valid_idx]
    y1_v = y1[valid_idx]

    z0 = (obj_vals[:, 0].reshape(1, -1) - y0_v) / s0_v
    z1 = (obj_vals[:, 1].reshape(1, -1) - y1_v) / s1_v

    z0 = np.clip(z0, -10, 10)
    z1 = np.clip(z1, -10, 10)

    from scipy.special import ndtr
    import scipy.stats as stats

    cdf_0 = ndtr(z0)
    cdf_1 = ndtr(z1)

    pdf_0 = stats.norm.pdf(z0)
    pdf_1 = stats.norm.pdf(z1)

    # compute PI vectorized for valid_idx directly
    pi_v = cdf_0[:, 0].copy()
    if obj_vals.shape[0] > 1:
        diff_cdf_0 = cdf_0[:, 1:] - cdf_0[:, :-1]
        # FIX: use cdf_1[:, :-1]
        pi_v += np.sum(diff_cdf_0 * cdf_1[:, :-1], axis=1)
    pi_v += (1.0 - cdf_0[:, -1]) * cdf_1[:, -1]

    pi_mask = pi_v > 1e-15
    if not np.any(pi_mask):
        return EI

    s0_pi = s0_v[pi_mask]
    s1_pi = s1_v[pi_mask]
    y0_pi = y0_v[pi_mask]
    y1_pi = y1_v[pi_mask]
    cdf_0_pi = cdf_0[pi_mask]
    cdf_1_pi = cdf_1[pi_mask]
    pdf_0_pi = pdf_0[pi_mask]
    pdf_1_pi = pdf_1[pi_mask]
    pi_final = pi_v[pi_mask]

    # Precompute integral bounds for Y0 and Y1
    # M0(b) = int_{-\infty}^b y_0 p(y_0) dy_0
    M0 = y0_pi * cdf_0_pi - s0_pi * pdf_0_pi
    # M1(b) = int_{-\infty}^b y_1 p(y_1) dy_1
    M1 = y1_pi * cdf_1_pi - s1_pi * pdf_1_pi

    # Y0 conditional expectation computation
    Y0 = M0[:, 0:1].copy()
    if obj_vals.shape[0] > 1:
        term1_0 = M0[:, 1:]
        term2_0 = M0[:, :-1]
        # FIX: use cdf_1_pi[:, :-1]
        Y0 += np.sum((term1_0 - term2_0) * cdf_1_pi[:, :-1], axis=1, keepdims=True)
    # FIX: Integral from f_0^{(n)} to infinity is (mu_0 - M0(f_0^{(n)}))
    Y0 += (y0_pi - M0[:, -1:]) * cdf_1_pi[:, -1:]
    Y0 /= pi_final.reshape(-1, 1)

    # Y1 conditional expectation computation
    # FIX: Complete rewrite of Y1 integration over non-dominated regions
    Y1 = cdf_0_pi[:, 0:1] * y1_pi
    if obj_vals.shape[0] > 1:
        diff_cdf_0_pi = cdf_0_pi[:, 1:] - cdf_0_pi[:, :-1]
        Y1 += np.sum(diff_cdf_0_pi * M1[:, :-1], axis=1, keepdims=True)
    Y1 += (1.0 - cdf_0_pi[:, -1:]) * M1[:, -1:]
    Y1 /= pi_final.reshape(-1, 1)

    ei_final = pi_final.reshape(-1, 1) * np.sqrt((Y0 - y0_pi) ** 2 + (Y1 - y1_pi) ** 2)

    final_indices = valid_idx[pi_mask]
    EI[final_indices] = ei_final

    return EI


def init_bi_obj_ei(state, kwargs=None):
    """
    Initialize the bi-objective Expected Improvement (EI) function.

    Parameters
    ----------
    state : State
        The current optimization state.
    kwargs : dict, optional
        Additional arguments, by default None.

    Returns
    -------
    Callable
        The initialized bi_obj_ei acquisition function.
    """

    pareto = get_pf_from_dataset(state.scaled_dataset)
    if pareto.shape[0] == 0:
        data = state.scaled_dataset.export_as_dict()
        min_rscv_idx = np.argmin(data["rscv"])
        pareto = data["obj"][min_rscv_idx, :].reshape(1, -1)

    # first objective needs to be sorted?
    sorted_idx = np.argsort(pareto[:, 0])
    sorted_pareto = pareto[sorted_idx, :]

    models = state.obj_models

    bi_obj_ei_func = partial(bi_obj_ei, obj_vals=sorted_pareto, models=models)

    return bi_obj_ei_func


def init_bi_obj_ei_cf(state, kwargs=None):
    """
    Initialize the Expected Improvement for Composite Functions (EI-CF).

    This function computes the Expected Improvement of a composite function
    using Monte Carlo integration.
    DOI: 10.48550/arXiv.1906.01537

    Parameters
    ----------
    state : State
        The current optimization state.
    kwargs : dict
        Arguments including 'phi' (the composite function) and 'n_accuracy' (number of MC samples).

    Returns
    -------
    Callable
        The initialized EI-CF acquisition function.
    """

    phi = kwargs["phi"]
    n_expectancy = kwargs.get("n_accuracy", 1000)

    models = state.obj_models
    data = state.scaled_dataset.export_as_dict()
    valid_mask = data["rscv"] <= 0.0
    if not np.any(valid_mask):
        valid_mask = data["rscv"] == np.min(data["rscv"])
    obj_vals = data["obj"][valid_mask, :]
    f_min = np.min(phi(obj_vals))

    # Pre-sample standard normal variables for Monte Carlo integration
    # This makes the acquisition function deterministic during optimization
    Z_fixed = np.random.randn(1, n_expectancy, 2)

    def ei_cf(x_pred: np.ndarray) -> np.ndarray:
        s0_sq = models[0].predict_variances(x_pred)
        s1_sq = models[1].predict_variances(x_pred)
        s = np.hstack(
            [np.sqrt(np.maximum(s0_sq, 0.0)), np.sqrt(np.maximum(s1_sq, 0.0))]
        )

        y0 = models[0].predict_values(x_pred)
        y1 = models[1].predict_values(x_pred)
        mu = np.hstack([y0, y1])

        samples = mu[:, None, :] + s[:, None, :] * Z_fixed
        phi_vals = phi(samples)
        ei = np.mean(np.maximum(f_min - phi_vals, 0.0), axis=1)
        return ei.reshape(-1, 1)

    return ei_cf


def init_bi_obj_ei_naive(state, kwargs=None):
    """
    Initialize the Expected Improvement using a Naive approach for Composite Functions.
    This trains a new surrogate model directly on the evaluated composite function values.
    """
    from smt_optim.acquisition_functions import log_ei
    import numpy as np

    phi = kwargs["phi"]
    data = state.scaled_dataset.export_as_dict()
    xt = data["x"]
    yt = data["obj"]
    y_phi = phi(yt).reshape(-1, 1)

    fidelity = data["fidelity"]
    xt_list = []
    y_phi_list = []
    for lvl in range(state.problem.num_fidelity):
        mask = (fidelity == lvl).ravel()
        xt_list.append(xt[mask, :])
        y_phi_list.append(y_phi[mask].reshape(-1, 1))

    kwargs_surrogate = state.problem.obj_configs[0].surrogate_kwargs
    if kwargs_surrogate is None:
        kwargs_surrogate = {}

    model = state.obj_models[0].__class__(
        design_space=state.problem.design_space, **kwargs_surrogate
    )
    model.train(xt_list, y_phi_list)

    valid_mask = data["rscv"] <= 0.0
    if not np.any(valid_mask):
        valid_mask = data["rscv"] == np.min(data["rscv"])
    f_min = np.min(y_phi[valid_mask])

    def ei_naive(x_pred: np.ndarray) -> np.ndarray:
        return log_ei(
            model.predict_values(x_pred), model.predict_variances(x_pred), f_min
        )

    return ei_naive
