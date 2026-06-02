from functools import partial
from typing import Callable

import numpy as np

import scipy.stats as stats

from smt_optim.utils.multi_obj import get_pf_from_dataset


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

    pareto_front = get_pf_from_dataset(state.scaled_dataset)         # shape: (n_points, n_objective)

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

