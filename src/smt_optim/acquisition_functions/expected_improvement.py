import numpy as np
import scipy.stats as stats
from scipy.special import erfcx
import copy


def expected_improvement(mu: float, s2: float, f_min: float) -> float:
    """
    Expected Improvement acquisition function.

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
        Expected Improvement value.
    """

    if s2 > 0:
        s = np.sqrt(s2)
        z = (f_min - mu)/s
        return (f_min - mu) * stats.norm.cdf(z) + s * stats.norm.pdf(z)
    else:
        return 0

def vec_expected_improvement(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:
    """
    Vectorized Expected Improvement acquisition function.

    Parameters
    ----------
    mu: np.ndarray
        Mean prediction of shape (num_points, 1).
    s2: np.ndarray
        Variance prediction of shape (num_points, 1).
    f_min: float
        Best minimum objective value in training data.

    Returns
    -------
    np.ndarray
        Expected Improvement values of shape (num_points, 1).
    """

    mask_s = s2 > 0
    ei = np.full_like(mu, 0)

    if not np.any(mask_s):
        return np.full_like(mu, 0)
    else:
        s = np.sqrt(s2[mask_s])
        z = (f_min - mu[mask_s])/s
        ei[mask_s] = (f_min - mu[mask_s])*stats.norm.cdf(z) + s*stats.norm.pdf(z)

    return ei



c1 = np.log(2*np.pi)/2
c2 = np.log(np.pi/2)/2
epsilon = np.finfo(np.float64).eps
sqrt2 = np.sqrt(2)


def log_ei(mu: float, s2: float, f_min: float) -> float:
    """
    Log Expected Improvement acquisition function.

    LogEI is more numerically stable that the EI acquisition function especially when the GP's variance is small.
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

    s = np.sqrt(s2)
    z = (f_min - mu)/s

    if z > -1:
        log_h = np.log(stats.norm.pdf(z) + z * stats.norm.cdf(z))

    elif (-1/np.sqrt(epsilon) < z) & (z <= -1):
        log_h = -z ** 2 / 2 - c1 + log1mexp(np.log(erfcx(-z / sqrt2) * np.abs(z)) + c2)

    elif z <= -1/np.sqrt(epsilon):
        log_h = -z**2/2 - c1 - 2*np.log(np.abs(z))

    else:
        raise Exception("Error computing LogEI")

    val = log_h + np.log(s)

    return val

def log1mexp(z: float) -> float:
    if z > -np.log(2):
        return np.log(-(np.expm1(z)))
    else:
        return np.log1p(-np.exp(z))


# ------- TODO: CLEAN LOG EXPECTED IMPROVEMENT -------
# ------- LOG EXPECTED IMPROVEMENT -------
def vec_log_ei(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:
    """
    Vectorized Log Expected Improvement acquisition function.

    LogEI is more numerically stable that the EI acquisition function especially when the GP's variance is small.
    From: https://arxiv.org/abs/2310.20708.

    Parameters
    ----------
    mu: np.ndarray
        Mean prediction of shape (num_points, 1).
    s2: np.ndarray
        Variance prediction of shape (num_points, 1).
    f_min: float
        Best minimum objective value in training data.

    Returns
    -------
    np.ndarray
    """
    s = np.sqrt(s2)

    c1 = np.log(2*np.pi)/2
    c2 = np.log(np.pi/2)/2
    epsilon = np.finfo(np.float64).eps

    z = np.zeros_like(mu)

    # create mask where std is equal or less than 0
    mask_s = (s > 0).ravel()
    not_mask_s = ~mask_s
    z[mask_s] = (f_min - mu[mask_s]) / s[mask_s]
    z[not_mask_s] = np.nan

    mask1 = (z > -1).ravel()
    mask2 = ((-1/np.sqrt(epsilon) < z) & (z <= -1)).ravel()
    mask3 = (z <= -1/np.sqrt(epsilon)).ravel()

    log_h = np.empty_like(z)

    log_h[mask1] = np.log(stats.norm.pdf(z[mask1]) + z[mask1] * stats.norm.cdf(z[mask1]))
    log_h[mask2] = -z[mask2]**2/2 - c1 + vec_log1mexp(np.log(erfcx(-z[mask2]/np.sqrt(2))*np.abs(z[mask2])) + c2)
    log_h[mask3] = -z[mask3]**2/2 - c1 - 2*np.log(np.abs(z[mask3]))

    log_ei = copy.deepcopy(log_h)
    log_ei[mask_s] = log_h[mask_s] + np.log(s[mask_s])

    # impose infinite value if std <= 0
    log_ei[not_mask_s] = -np.inf

    return log_ei


def vec_log1mexp(z: np.ndarray) -> np.ndarray:
    # review how mask1 is computed -> see not regular implementation
    mask1 = (-2*np.log(2) < z).ravel()
    mask2 = ~mask1

    log1mexp_arr = np.empty_like(z)

    log1mexp_arr[mask1] = np.log(-(np.exp(z[mask1]) - 1))
    log1mexp_arr[mask2] = np.log1p(-np.exp(z[mask2]))

    return log1mexp_arr








