from functools import partial
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist

from moocore import hypervolume as moocore_hv


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


def get_pf_from_dataset(dataset, ctol: float = 1e-4, fid: int = -1) -> np.ndarray:

    data_dict = dataset.export_as_dict()
    obj = data_dict["obj"]
    rscv = data_dict["rscv"]
    fidelity = data_dict["fidelity"]

    feas_mask = (rscv <= ctol)

    if fid == -1:
        fid = np.max(fidelity)

    fid_mask = fidelity[feas_mask] == fid

    pareto_front = get_pareto_front(obj[feas_mask][fid_mask])

    return pareto_front



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


def hypervolume(pf: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the hypervolume indicator of the Pareto front.

    Uses the `moocore` implementation:
    https://multi-objective.github.io/moocore/python/reference/generated/moocore.hypervolume.html

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

    return moocore_hv(pf, ref=ref)


def spacing(pf: np.ndarray) -> float:
    """
    Compute the spacing indicator of the Pareto front (Schott, 1995). A lower value is better.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, num_objectives)
        Pareto front.

    Returns
    -------
    float
        Spacing indicator value.

    Notes:
        Assume both objective are minimized.
    """

    num_pf = pf.shape[0]

    distances = cdist(pf, pf, "cityblock")
    np.fill_diagonal(distances, np.inf)

    d1 = np.min(distances, axis=1)
    d1_mean = np.mean(d1)

    value = np.sqrt(1/(num_pf-1) * np.sum((d1_mean - d1) ** 2))

    return value