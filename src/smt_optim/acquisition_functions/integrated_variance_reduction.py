# @autor Paul-Saves
from typing import Callable

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular
from smt.utils.kriging import cross_distances, differences


def _extract_smt_params(model):
    """
    Extract the SMT Kriging/MFK internals required by the look-ahead variance.

    Parameters
    ----------
    model : Surrogate
        smt-optim surrogate model (must wrap an SMT Kriging/MFK model).

    Returns
    -------
    tuple
        (smt_model, sigma2, theta, X_train, nt, Cn, nugget, X_offset, X_scale), where X_train
        are the (high-fidelity) training points and Cn the Cholesky factor of their correlation
        matrix.

    Raises
    ------
    TypeError
        If the model is not an SMT Kriging/MFK model.
    """
    smt_model = getattr(model, "model", model)

    if not hasattr(smt_model, "optimal_par"):
        raise TypeError(
            f"IVR requires an SMT Kriging/MFK surrogate model with 'optimal_par', "
            f"got {type(smt_model).__name__}."
        )

    if hasattr(smt_model, "nlvl"):  # MFK: only the high-fidelity level
        sigma2 = smt_model.optimal_par[-1]["sigma2"]
        theta = smt_model.optimal_theta[-1]
        Cn = smt_model.optimal_par[-1]["C"]
    else:  # KRG
        sigma2 = smt_model.optimal_par["sigma2"]
        theta = smt_model.optimal_theta
        Cn = smt_model.optimal_par["C"]

    X_train = smt_model.training_points[None][0][0]

    try:
        nugget = smt_model.options["nugget"]
    except KeyError:
        nugget = 1e-10

    return (
        smt_model,
        sigma2,
        theta,
        X_train,
        len(X_train),
        Cn,
        nugget,
        smt_model.X_offset,
        smt_model.X_scale,
    )


def variance_update(model, point, x, inv_block=True):
    """
    Compute the "look-ahead" conditional variance after adding the new point to the model.

    Parameters
    ----------
    model : Surrogate
        smt-optim surrogate model (must wrap an SMT Kriging/MFK model).
    point : np.ndarray
        Enrichment point (candidate).
    x : np.ndarray()
        Evaluation points (integration grid).
    inv_block : bool, optional
        Compute the augmented correlation matrix using block inversion. Default True.

    Returns
    -------
    MSE : np.ndarray
        Look-ahead variance at evaluation points.
    """
    smt_model, sigma2, theta, X_train, nt, Cn, nugget, X_offset, X_scale = (
        _extract_smt_params(model)
    )
    n_eval = x.shape[0]
    X_train = np.vstack([X_train, point])

    # Standardization of data using the model's exact scaling
    X_normalized = (X_train - X_offset) / X_scale
    x_normalized = (x - X_offset) / X_scale

    # Compute componentwise distance
    dx = differences(x_normalized, X_normalized)
    d = smt_model._componentwise_distance(dx)

    # Compute the correlation function
    smt_model.corr.theta = theta
    r = smt_model.corr(d).reshape(n_eval, nt + 1)

    if inv_block:
        point_norm = X_normalized[nt, :].reshape(1, -1)
        doe_normalized = X_normalized[0:nt, :]
        dx_nu = differences(point_norm, doe_normalized)
        dnu = smt_model._componentwise_distance(dx_nu)
        smt_model.corr.theta = theta
        rn_nu = smt_model.corr(dnu)

        v = solve_triangular(Cn, rn_nu, lower=True)

        c2 = (1.0 + nugget) - np.sum(v**2)
        if c2 <= 1e-6:
            # The point is already in the dataset (or numerically indistinguishable)
            # Its variance reduction is 0, so the variance remains sigma2
            return current_variance(model, x)

        c = np.sqrt(c2)

        C_aug = np.zeros((nt + 1, nt + 1))
        C_aug[:nt, :nt] = Cn
        C_aug[nt, :nt] = v.flatten()
        C_aug[nt, nt] = c

        W = solve_triangular(C_aug, r.T, lower=True)
        res = np.sum(W**2, axis=0)
    else:
        # Full inversion
        dist, ij = cross_distances(X_normalized)
        D = smt_model._componentwise_distance(dist)
        smt_model.corr.theta = theta
        r_ = smt_model.corr(D)

        R = np.eye(nt + 1) * (1 + nugget)
        R[ij[:, 0], ij[:, 1]] = r_[:, 0]
        R[ij[:, 1], ij[:, 0]] = r_[:, 0]

        try:
            C_aug = np.linalg.cholesky(R)
            W = solve_triangular(C_aug, r.T, lower=True)
            res = np.sum(W**2, axis=0)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
            res = np.sum((r @ R_inv) * r, axis=1)

    # Compute the "look-ahead" conditional variance
    MSE = sigma2 * (1 - res)
    return np.maximum(MSE, 0.0)


def current_variance(model, x):
    """
    Compute the conditional variance of the current model.
    """
    smt_model, sigma2, theta, X_train, nt, Cn, _, X_offset, X_scale = (
        _extract_smt_params(model)
    )
    n_eval = x.shape[0]

    X_normalized = (X_train - X_offset) / X_scale
    x_normalized = (x - X_offset) / X_scale

    dx = differences(x_normalized, X_normalized)
    d = smt_model._componentwise_distance(dx)
    smt_model.corr.theta = theta
    r = smt_model.corr(d).reshape(n_eval, nt)

    W = solve_triangular(Cn, r.T, lower=True)
    res = np.sum(W**2, axis=0)
    MSE = sigma2 * (1 - res)
    return np.maximum(MSE, 0.0)


def _prepare_ivr_grid(model, integration_points: np.ndarray) -> dict:
    """
    Precompute the terms of the IVR that do not depend on the candidate points.

    Parameters
    ----------
    model : Surrogate
        smt-optim surrogate model (must wrap an SMT Kriging/MFK model).
    integration_points : np.ndarray
        Monte-Carlo points for integration over the domain, shape (N_mc, num_dim).

    Returns
    -------
    dict
        Model internals, normalized points and the current variance on the integration grid.
    """
    smt_model, sigma2, theta, X_train, nt, Cn, nugget, X_offset, X_scale = (
        _extract_smt_params(model)
    )

    n_eval = integration_points.shape[0]
    x_train_norm = (X_train - X_offset) / X_scale
    eval_norm = (integration_points - X_offset) / X_scale

    smt_model.corr.theta = theta
    r_base = smt_model.corr(
        smt_model._componentwise_distance(differences(eval_norm, x_train_norm))
    ).reshape(n_eval, nt)

    W_base = solve_triangular(Cn, r_base.T, lower=True)
    mse_base_raw = sigma2 * (1.0 - np.sum(W_base**2, axis=0))
    imse_current = float(np.mean(np.maximum(mse_base_raw, 0.0)))

    return {
        "smt_model": smt_model,
        "sigma2": sigma2,
        "theta": theta,
        "nt": nt,
        "Cn": Cn,
        "nugget": nugget,
        "X_offset": X_offset,
        "X_scale": X_scale,
        "x_train_norm": x_train_norm,
        "eval_norm": eval_norm,
        "W_base": W_base,
        "mse_base_raw": mse_base_raw,
        "imse_current": imse_current,
    }


def _eval_ivr_from_grid(grid_cache: dict, points: np.ndarray) -> np.ndarray:
    """
    Evaluate the IVR of several candidate points at once from the precomputed grid terms.

    Parameters
    ----------
    grid_cache : dict
        Output of `_prepare_ivr_grid`.
    points : np.ndarray
        Candidate points, shape (num_points, num_dim).

    Returns
    -------
    np.ndarray
        The IVR values of shape (num_points, 1).
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)

    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros((0, 1))

    smt_model = grid_cache["smt_model"]
    eval_norm = grid_cache["eval_norm"]
    n_eval = eval_norm.shape[0]
    nt = grid_cache["nt"]

    points_norm = (points - grid_cache["X_offset"]) / grid_cache["X_scale"]

    smt_model.corr.theta = grid_cache["theta"]
    r_nu = smt_model.corr(
        smt_model._componentwise_distance(
            differences(points_norm, grid_cache["x_train_norm"])
        )
    ).reshape(num_points, nt)
    V = solve_triangular(grid_cache["Cn"], r_nu.T, lower=True)

    c2 = (1.0 + grid_cache["nugget"]) - np.sum(V**2, axis=0)
    r_ce = smt_model.corr(
        smt_model._componentwise_distance(differences(points_norm, eval_norm))
    ).reshape(num_points, n_eval)

    # a candidate already in the dataset (c2 <= 1e-6) does not reduce the variance
    valid = c2 > 1e-6
    safe_c2 = np.where(valid, c2, 1.0)
    w_last_sq = ((r_ce - V.T @ grid_cache["W_base"]) ** 2) / safe_c2[:, None]

    mse_new = np.maximum(
        grid_cache["mse_base_raw"][None, :] - grid_cache["sigma2"] * w_last_sq, 0.0
    )
    ivr_vals = np.where(
        valid, grid_cache["imse_current"] - np.mean(mse_new, axis=1), 0.0
    )
    return ivr_vals[:, None]


def integrated_variance_reduction(
    model,
    points: np.ndarray,
    integration_points: np.ndarray = None,
    inv_block: bool = True,
) -> np.ndarray:
    """
    Integrated Variance Reduction (IVR) acquisition function.
    Evaluates IVR for one or multiple candidate points.
    Returns the absolute reduction in IMSE: |IMSE_current - IMSE_new|.

    Parameters
    ----------
    model : Surrogate
        The smt-optim surrogate model representing the objective function.
    points : np.ndarray
        Candidate points for enrichment, shape (num_points, num_dim).
    integration_points : np.ndarray, optional
        Monte-Carlo points for integration over the domain, shape (N_mc, num_dim).
        If None, a 500-point LHS grid is automatically generated (requires model to have xlimits or infers from training points).
    inv_block : bool, optional
        Whether to use the vectorized rank-1 update of the Cholesky factor (faster). If False,
        the augmented correlation matrix of each candidate is factorized. Default True.

    Returns
    -------
    np.ndarray
        The IVR values of shape (num_points, 1) (higher is better).
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)

    smt_model = getattr(model, "model", model)

    if integration_points is None:
        if hasattr(smt_model, "_default_integration_points"):
            integration_points = smt_model._default_integration_points
        else:
            from smt.sampling_methods import LHS

            try:
                xlimits = smt_model.options["xlimits"]
            except KeyError:
                xlimits = None

            if xlimits is None:
                X_train = smt_model.training_points[None][0][0]

                xlimits = np.vstack(
                    (np.min(X_train, axis=0), np.max(X_train, axis=0))
                ).T
                import warnings

                warnings.warn(
                    "integration_points was not provided and model does not have 'xlimits'. "
                    "Inferring domain bounds from training data, which may lead to inaccurate integration."
                )

            integration_points = LHS(xlimits=xlimits, criterion="ese", seed=42)(500)
            smt_model._default_integration_points = integration_points

    if inv_block:
        # vectorized rank-1 update of the Cholesky factor
        return _eval_ivr_from_grid(_prepare_ivr_grid(model, integration_points), points)

    num_points = points.shape[0]
    ivr_vals = np.zeros((num_points, 1))

    # Compute current IMSE once
    imse_current = np.mean(current_variance(model, integration_points))

    for i in range(num_points):
        MSE_new = variance_update(
            model, points[i : i + 1, :], integration_points, inv_block=False
        )
        imse_new = np.mean(MSE_new)
        # We want to maximize the reduction (IMSE_current - IMSE_new).
        # We take the absolute value as requested, though mathematically imse_current >= imse_new.
        ivr_vals[i, 0] = imse_current - imse_new

    return ivr_vals


def init_ivr(state, n_integration: int = 64) -> Callable:
    """
    Initialize the Integrated Variance Reduction (IVR) acquisition function of the objective
    surrogate model.

    Parameters
    ----------
    state : State
        Optimization state containing the problem definition and the trained objective
        surrogate model.
    n_integration : int, optional
        Number of Sobol integration points. Default is 64.

    Returns
    -------
    Callable
        IVR acquisition function with signature::

            acquisition(x) -> np.ndarray

        where `x` is of shape (num_points, num_dim) and the returned values are of shape
        (num_points, 1). Higher is better.

    Notes
    -----
    The integration points are sampled once in the unit hypercube, so that all the evaluations
    of an iteration share the same integration grid, and the terms that do not depend on the
    candidate points are computed once. `x` must be in the scaled design space.
    """

    if n_integration < 1:
        raise ValueError(f"n_integration must be at least 1, got {n_integration}.")

    model = state.obj_models[0]

    sampler = stats.qmc.Sobol(d=state.problem.num_dim, seed=state.iter)
    integration_points = sampler.random(n_integration)

    grid_cache = _prepare_ivr_grid(model, integration_points)

    def ivr_func(x: np.ndarray) -> np.ndarray:
        return _eval_ivr_from_grid(grid_cache, x)

    return ivr_func
