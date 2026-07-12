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

    pareto_front = get_pf_from_dataset(
        state.scaled_dataset
    )  # shape: (n_points, n_objective)

    # if no feasible point in pareto front (possible in constrained optimization)
    if pareto_front.shape[0] == 0:
        data = state.scaled_dataset.export_as_dict()
        min_rscv_idx = np.argmin(data["rscv"])
        pareto_front = data["obj"][min_rscv_idx, :].reshape(1, -1)

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
            s2_obj = state.obj_models[idx].predict_variances(x).item()

            if s2_obj <= 1e-12:
                return 1.0

            s_obj = np.sqrt(np.maximum(s2_obj, 0.0))

            values *= stats.norm.cdf((mu_obj - pareto_front[:, idx]) / s_obj)

        return 1 - np.max(values)

    return mpi_func
