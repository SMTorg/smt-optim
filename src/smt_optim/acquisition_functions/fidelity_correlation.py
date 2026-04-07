import numpy as np

def fidelity_correlation(covariance: np.ndarray, li_var: np.ndarray, lj_var: np.ndarray) -> np.ndarray:
    """
    Compute the posterior correlation between two fidelity levels.

    This function evaluates the pointwise Pearson correlation coefficient
    between the predictive distributions of two fidelity levels based on
    their posterior covariance and variances. The resulting correlation
    is clipped to the interval [0, 1].

    Parameters
    ----------
    covariance : ndarray of shape (n_eval,)
        Posterior covariance between the predictions at fidelity levels
        :math:`i` and :math:`j` for each evaluation point.
    li_var : ndarray of shape (n_eval,)
        Posterior predictive variance at fidelity level :math:`i`.
    lj_var : ndarray of shape (n_eval,)
        Posterior predictive variance at fidelity level :math:`j`.

    Returns
    -------
    ndarray of shape (n_eval,)
        Absolute value of the correlation coefficient between fidelity
        levels :math:`i` and :math:`j`, clipped to the interval [0, 1].
    """
    return np.clip(np.abs(covariance/np.sqrt(li_var * lj_var)), 0, 1)