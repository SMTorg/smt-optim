import numpy as np
from scipy.special import logsumexp


def q_log_ei(
    mu: np.ndarray,
    cov: np.ndarray,
    f_min: float,
    z: np.ndarray,
    tau_relu: float = 1e-6,
    tau_max: float = 1e-2,
    nugget: float = 1e-9,
) -> float:
    """
    q-Log Expected Improvement (qLogEI) acquisition function.

    From: https://arxiv.org/abs/2310.20708.

    Parameters
    ----------
    mu : np.ndarray
        Mean prediction of the batch of shape (batch_size, 1).
    cov : np.ndarray
        Covariance prediction of the batch of shape (batch_size, batch_size).
    f_min : float
        Best minimum objective value in training data.
    z : np.ndarray
        Standard normal samples of shape (n_samples, batch_size). Keep them fixed to obtain a
        deterministic function of the batch.
    tau_relu : float, optional
        Smoothing parameter of the improvement. Default is 1e-6.
    tau_max : float, optional
        Smoothing parameter of the maximum over the batch. Default is 1e-2.
    nugget : float, optional
        Value added to the diagonal of the covariance matrix for numerical stability. Default
        is 1e-9.

    Returns
    -------
    float
        qLogEI value of the batch. Higher is better.
    """

    # scenarios of shape (n_samples, batch_size)
    L = np.linalg.cholesky(cov + nugget * np.eye(cov.shape[0]))
    y = mu.ravel() + z @ L.T

    # log of the smoothed improvement max(f_min - y, 0) of each point
    scaled_improvement = (f_min - y) / tau_relu
    log_improvement = np.log(
        tau_relu
        * (np.logaddexp(0.0, scaled_improvement) + 0.1 / (1.0 + scaled_improvement**2))
    )

    # smoothed maximum over the batch of each scenario
    log_best = np.max(log_improvement, axis=1, keepdims=True)
    gap = (log_best - log_improvement) / tau_max
    weights = 2.0 / (2.0 + 2.0 * gap + gap**2)
    log_best_improvement = log_best.ravel() + tau_max * np.log(np.sum(weights, axis=1))

    # log of the mean over the scenarios
    return float(logsumexp(log_best_improvement) - np.log(z.shape[0]))
