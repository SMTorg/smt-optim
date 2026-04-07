import numpy as np
import scipy.stats as stats

def probability_of_improvement(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:
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