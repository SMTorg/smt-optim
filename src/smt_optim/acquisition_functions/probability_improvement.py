import numpy as np
import scipy.stats as stats
from scipy.special import erfc, erfcx


def probability_of_improvement(mu: float, s2: float, f_min: float) -> float:
    """
    Probability of Improvement (PI) acquisition function.

    Parameters
    ----------
    mu: float
        Mean prediction.
    s2: float
        Variance prediction.
    f_min: float
        Best minimum objective value in training data.

    Returns
    -------
    float
        Probability of Improvement value.
    """
    if s2 <= 0:
        return 0

    pi = stats.norm.cdf((f_min - mu)/np.sqrt(s2))
    return pi


def vec_probability_of_improvement(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition function.

    Parameters
    ----------
    mu: float
        Mean prediction.
    s2: float
        Variance prediction.
    f_min: float
        Best minimum objective value in training data.

    Returns
    -------
    float
        Probability of Improvement value.
    """
    pi = np.zeros_like(mu)

    mask_s = (s2 > 0)
    pi[mask_s] = stats.norm.cdf((f_min - mu[mask_s])/np.sqrt(s2[mask_s]))
    return pi


def log_pi(mu: float, s2: float, f_min: float) -> float:
    """
    Log Probability of Improvement acquisition function.

    LogPI is more numerically stable that the pi acquisition function especially when the GP's variance is small.
    From: https://arxiv.org/abs/2310.20708.

    Parameters
    ----------
    mu: float
        Mean prediction
    s2: float
        Variance prediction
    f_min: float
        Best minimum objective value in training data.

    Returns
    -------
    float
    """

    if s2 <= 0:
        return -np.inf

    z = (f_min - mu)/np.sqrt(s2)

    return logerfc(-1/np.sqrt(2) * z) - np.log(2)


def logerfc(x: float) -> float:
    if x <= 0:
        return np.log(erfc(x))
    else:
        return np.log(erfcx(x)) - x**2


