#@autor Paul-Saves
import numpy as np
from scipy.linalg import solve_triangular
from smt.utils.kriging import cross_distances, differences
from smt.utils.misc import standardization

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
        Evaluation points (Monte-Carlo integration grid).
    inv_block : bool, optional
        Compute the augmented correlation matrix using block inversion. Default True.

    Returns
    -------
    MSE : np.ndarray
        Look-ahead variance at evaluation points.
    """
    smt_model = getattr(model, "model", model)
    n_eval = x.shape[0]
    ndim = point.shape[1]

    # Handle Multi-Fidelity (MFK) vs Mono-Fidelity (KRG)
    if hasattr(smt_model, "nlvl"):  # MFK
        sigma2 = smt_model.optimal_par[-1]["sigma2"]
        theta = smt_model.optimal_theta[-1]
        # Only consider HF points for the model base
        nt = smt_model.nt
        # In MFK, training points are grouped by level. HF is None.
        X_train_hf = smt_model.training_points[None][0][0]
        nt = len(X_train_hf)
        X_train = np.vstack([X_train_hf, point])
        Cn = smt_model.optimal_par[-1]["C"]
    else:  # KRG
        sigma2 = smt_model.optimal_par["sigma2"]
        theta = smt_model.optimal_theta
        X_train_mono = smt_model.training_points[None][0][0]
        X_train = np.vstack([X_train_mono, point])
        nt = smt_model.nt
        Cn = smt_model.optimal_par["C"]

    # Standardization of data using the model's exact scaling
    X_offset = smt_model.X_offset
    X_scale = smt_model.X_scale
    
    X_normalized = (X_train - X_offset) / X_scale
    x_normalized = (x - X_offset) / X_scale

    # Compute componentwise distance
    dx = differences(x_normalized, X_normalized)
    d = smt_model._componentwise_distance(dx)

    # Compute the correlation function
    smt_model.corr.theta = theta
    r = smt_model.corr(d).reshape(n_eval, nt + 1)

    try:
        nugget = smt_model.options["nugget"]
    except KeyError:
        nugget = 1e-10

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
        dist, ij = cross_distances(X_train)
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
    Compute the conditional variance of the current model (without new points).
    """
    smt_model = getattr(model, "model", model)
    n_eval = x.shape[0]

    if hasattr(smt_model, "nlvl"):  # MFK
        sigma2 = smt_model.optimal_par[-1]["sigma2"]
        theta = smt_model.optimal_theta[-1]
        nt = smt_model.nt
        X_train = smt_model.training_points[None][0][0]
        Cn = smt_model.optimal_par[-1]["C"]
    else:  # KRG
        sigma2 = smt_model.optimal_par["sigma2"]
        theta = smt_model.optimal_theta
        X_train = smt_model.training_points[None][0][0]
        nt = smt_model.nt
        Cn = smt_model.optimal_par["C"]

    X_offset = smt_model.X_offset
    X_scale = smt_model.X_scale
    
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


def integrated_variance_reduction(model, points: np.ndarray, integration_points: np.ndarray = None, inv_block: bool = True) -> np.ndarray:
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
        Whether to use block matrix inversion (faster). Default True.
        
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
            xlimits = smt_model.options.get("xlimits")
            
            if xlimits is None:
                if hasattr(smt_model, "nlvl"):
                    X_train = smt_model.training_points[None][0][0]
                else:
                    X_train = smt_model.training_points[None][0][0]
                
                xlimits = np.vstack((np.min(X_train, axis=0), np.max(X_train, axis=0))).T
                import warnings
                warnings.warn(
                    "integration_points was not provided and model does not have 'xlimits'. "
                    "Inferring domain bounds from training data, which may lead to inaccurate integration."
                )
                
            integration_points = LHS(xlimits=xlimits, criterion='ese', seed=42)(500)
            smt_model._default_integration_points = integration_points
        
    num_points = points.shape[0]
    ivr_vals = np.zeros((num_points, 1))
    
    # Compute current IMSE once
    imse_current = np.mean(current_variance(model, integration_points))
    
    for i in range(num_points):
        MSE_new = variance_update(model, points[i:i+1, :], integration_points, inv_block=inv_block)
        imse_new = np.mean(MSE_new)
        # We want to maximize the reduction (IMSE_current - IMSE_new).
        # We take the absolute value as requested, though mathematically imse_current >= imse_new.
        ivr_vals[i, 0] = imse_current - imse_new
        
    return ivr_vals
