import numpy as np
import scipy.stats as stats
from scipy.special import erfcx
import copy

from multificbo.surrogate_models import Surrogate

def expected_improvement(mu, s2, f_min):
    mask_s = (s2 > 0).ravel()

    s = np.empty_like(s2)
    s[mask_s] = np.sqrt(s2[mask_s])

    ei = np.full_like(mu, -np.inf)
    z = np.empty_like(mu)

    z[mask_s] = (f_min - mu[mask_s])/s[mask_s]
    ei[mask_s] = (f_min - mu[mask_s])*stats.norm.cdf(z[mask_s]) + s[mask_s]*stats.norm.pdf(z[mask_s])

    return ei

# ------- TODO: CLEAN LOG EXPECTED IMPROVEMENT -------
# ------- LOG EXPECTED IMPROVEMENT -------
def log_ei(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:

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
    log_h[mask2] = -z[mask2]**2/2 - c1 + log1mexp(np.log(erfcx(-z[mask2]/np.sqrt(2))*np.abs(z[mask2])) + c2)
    log_h[mask3] = -z[mask3]**2/2 - c1 - 2*np.log(np.abs(z[mask3]))

    log_ei = copy.deepcopy(log_h)
    log_ei[mask_s] = log_h[mask_s] + np.log(s[mask_s])

    # impose infinite value if std <= 0
    log_ei[not_mask_s] = -np.inf

    return log_ei

def log1mexp(z: np.ndarray) -> np.ndarray:

    mask1 = (-2*np.log(2) < z).ravel()
    mask2 = ~mask1

    log1mexp_arr = np.empty_like(z)

    log1mexp_arr[mask1] = np.log(-(np.exp(z[mask1]) - 1))
    log1mexp_arr[mask2] = np.log1p(-np.exp(z[mask2]))

    return log1mexp_arr


def fidelity_correlation(covariance: np.ndarray, li_var: np.ndarray, lj_var: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(covariance/np.sqrt(li_var * lj_var)), 0, 1)


def probability_of_improvement(mu: np.ndarray, s2: np.ndarray, f_min: float) -> np.ndarray:

    pi = np.zeros_like(mu)

    mask_s = (s2 > 0)
    pi[mask_s] = stats.norm.cdf((f_min - mu[mask_s])/np.sqrt(s2[mask_s]))
    return pi





