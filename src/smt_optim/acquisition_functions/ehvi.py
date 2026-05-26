from functools import partial
from typing import Callable

import numpy as np

import scipy.stats as stats

from smt_optim.utils.multi_obj import get_pf_from_dataset


def psi(
    a: float | np.ndarray,
    b: float | np.ndarray,
    mu: float | np.ndarray,
    s: float | np.ndarray,
) -> float | np.ndarray:
    """
    Helper function to be used with the EHVI (`ehvi_2o`) acquisition function.

    Parameters
    ----------
    a : float or np.array
    b : float or np.array
    mu : float or np.array
    s : float or np.array

    Returns
    -------
    float or np.ndarray
    """
    z = (b - mu) / s
    return s * stats.norm.pdf(z) + (a - mu) * stats.norm.cdf(z)


def ehvi_2o(mu: np.ndarray, s: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Expected Hypervolume Improvement (EHVI) for bi-objective optimization.

    Parameters
    ----------
    mu : np.ndarray of shape (2,)
        Predictive mean of the two objective functions.

    s : np.ndarray of shape (2,)
        Predictive standard deviation of the two objective functions.

    Y : np.ndarray of shape (n, 2)
        Augmented Pareto front, sorted in ascending order with respect to the
        second objective.

    Returns
    -------
    float
        The expected hypervolume improvement.

    Notes
    -----
    Follows the implementation discussed in:
    Yang, K., Emmerich, M., Deutz, A., & Bäck, T. (2019). Multi-objective Bayesian global optimization using
    expected hypervolume improvement gradient. Swarm and evolutionary computation, 44, 945-956.
    """
    # term1 is truncated because: np.inf * 0. = np.nan
    term1 = (
        (Y[:-1, 0] - Y[1:, 0])[:-1]
        * stats.norm.cdf((Y[1:, 0] - mu[0]) / s[0])[:-1]
        * psi(Y[1:, 1], Y[1:, 1], mu[1], s[1])[:-1]
    )
    term2 = (
        psi(Y[:-1, 0], Y[:-1, 0], mu[0], s[0]) - psi(Y[:-1, 0], Y[1:, 0], mu[0], s[0])
    ) * psi(Y[1:, 1], Y[1:, 1], mu[1], s[1])
    return term1.sum() + term2.sum()


def init_ehvi_2o(state) -> Callable:
    """
    Initialize the Expected Hypervolume Improvement (EHVI) acquisition function
    for bi-objective Bayesian optimization.

    The returned callable evaluates EHVI using the objective surrogate models
    stored in `state`. The Pareto front is extracted from the scaled dataset
    and augmented with reference points required by the two-objective EHVI
    computation (`ehvi_2o`).

    Parameters
    ----------
    state : State
        Optimization state containing the problem definition, scaled dataset,
        and trained objective surrogate models.

        The problem must have exactly two objectives. The scaled dataset must
        contain objective evaluations from which the current Pareto front can
        be computed.

    Returns
    -------
    Callable
        EHVI acquisition function with signature::

            acquisition(x) -> float

        where `x` is a candidate point and the returned value is the expected
        hypervolume improvement at `x`.

    Raises
    ------
    ValueError
        If the optimization problem is not bi-objective.

    Notes
    -----
    The Pareto front is augmented with two additional points and sorted by the
    second objective before being passed to the EHVI computation.

    Follows the implementation discussed in:
    Yang, K., Emmerich, M., Deutz, A., & Bäck, T. (2019). Multi-objective Bayesian global optimization using
    expected hypervolume improvement gradient. Swarm and evolutionary computation, 44, 945-956.
    """
    if state.problem.num_obj != 2:
        raise ValueError(
            f"EHVI (2-objective) only supports bi-objective optimization, but got "
            f"{state.problem.num_obj} objectives. Use a compatible acquisition function."
        )

    data = state.scaled_dataset.export_as_dict()
    obj = data["obj"]
    pf = get_pf_from_dataset(state.scaled_dataset)

    r1 = obj[:, 0].max()
    r2 = obj[:, 1].max()

    # augment PF by 2 points
    augmented_pf = np.vstack((np.array([[r1, -np.inf]]), pf, np.array([[-np.inf, r2]])))

    # sort augmented PF by the second objective
    indices = np.argsort(augmented_pf[:, 1])
    Y = augmented_pf[indices, :]

    def ehvi_wrapper(x: np.ndarray, models: list, Y: np.ndarray) -> float:

        mu = np.empty(2)
        s = np.empty(2)

        for i in range(2):
            mu[i] = models[i].predict_values(x).item()
            s[i] = np.sqrt(models[i].predict_variances(x).item())

        return ehvi_2o(mu, s, Y)

    return partial(ehvi_wrapper, models=state.obj_models, Y=Y)
