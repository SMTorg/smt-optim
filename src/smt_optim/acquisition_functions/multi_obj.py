from functools import partial
from typing import Callable

import numpy as np

import scipy.stats as stats


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

    data_dict = state.scaled_dataset.export_as_dict()
    obj = data_dict["obj"]
    rscv = data_dict["rscv"]

    feas_mask = (rscv <= 1e-4)

    pareto_front = get_pareto_front(obj[feas_mask])         # shape: (n_points, n_objective)

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
            s_obj = np.sqrt(state.obj_models[idx].predict_variances(x).item())

            values *= stats.norm.cdf((mu_obj - pareto_front[:, idx])/s_obj)

        return 1 - np.max(values)

    return mpi_func



def get_pareto_mask(Y: np.ndarray) -> np.ndarray:

    n = Y.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        # A point is dominated if another point is <= in all objectives
        # and strictly < in at least one
        dominates = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)

        # If any point dominates i -> i is not Pareto
        if np.any(dominates):
            is_pareto[i] = False

    return is_pareto


def get_pareto_front(Y: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated objective vectors from ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Array of shape ``(n_samples, n_objectives)`` containing objective
        values for each sample.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_pareto, n_objectives)`` containing the
        non-dominated objective vectors (the Pareto front).

    Notes
    -----
    Assumes a minimization problem for all objectives and no constraints.
    """

    pareto_mask = get_pareto_mask(Y)
    pareto = Y[pareto_mask]
    return pareto


def hypervolume_2d(pf: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the 2D hypervolume indicator  he hypervolume of the Pareto front.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, 2)
        Pareto front.

    ref: np.ndarray of shape (2, )
        Reference objective values.

    Returns
    -------
    float
        Hypervolume indicator value.

    Notes:
        Assume both objective are minimized.
    """

    if pf.shape[1] != 2 or ref.shape[0] != 2:
        raise Exception("Current hypervolume implementation is only for bi-objective optimization.")

    sorted_idx = np.argsort(pf[:, 0])
    sorted_pf = pf[sorted_idx]

    hv = 0.
    prev_f2 = ref[1]

    for idx in range(sorted_pf.shape[0]):

        f1 = sorted_pf[idx, 0]
        f2 = sorted_pf[idx, 1]

        width = ref[0] - f1
        height = prev_f2 - f2

        if width > 0 and height > 0:
            hv += width * height

        prev_f2 = min(prev_f2, f2)

    return hv

