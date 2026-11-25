from typing import Callable

import numpy as np


def init_mean(state) -> Callable:
    """
    Initialize the mean prediction acquisition function of the objective surrogate model.

    Parameters
    ----------
    state : State
        Optimization state containing the trained objective surrogate model.

    Returns
    -------
    Callable
        Mean prediction acquisition function with signature::

            acquisition(x) -> np.ndarray

        where `x` is of shape (num_points, num_dim) and the returned values are of shape
        (num_points, 1).
    """

    model = state.obj_models[0]

    def mean_func(x: np.ndarray) -> np.ndarray:
        return model.predict_values(x)

    return mean_func
