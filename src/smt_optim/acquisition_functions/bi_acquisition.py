from typing import Callable

import numpy as np


def negate(acq_init: Callable) -> Callable:
    """
    Negate an acquisition function initializer, so that an acquisition function to be maximized
    can be minimized.

    Parameters
    ----------
    acq_init : Callable
        Initializer of the acquisition function, with signature::

            acq_init(state) -> acquisition

    Returns
    -------
    Callable
        Initializer of the negated acquisition function, with the same signature as `acq_init`.
    """

    def neg_acq_init(state) -> Callable:
        acq = acq_init(state)

        def neg_acq(x: np.ndarray) -> np.ndarray:
            return -acq(x)

        return neg_acq

    return neg_acq_init


def init_bi_acquisition(state, acq_init_1: Callable, acq_init_2: Callable) -> Callable:
    """
    Initialize the bi-objective acquisition function built from two acquisition functions.

    The returned callable evaluates both acquisition functions at the same points. The resulting
    bi-objective problem is a minimization problem.

    Parameters
    ----------
    state : State
        Optimization state containing the problem definition and trained surrogate models.
    acq_init_1 : Callable
        Initializer of the first acquisition function, with signature::

            acq_init_1(state) -> acquisition

        where `acquisition(x)` takes points of shape (num_points, num_dim) and returns one value
        per point, to be minimized.
    acq_init_2 : Callable
        Initializer of the second acquisition function, with the same signature as `acq_init_1`.

    Returns
    -------
    Callable
        Bi-objective acquisition function with signature::

            acquisition(x) -> np.ndarray

        where `x` is of shape (num_points, num_dim) and the returned values are of shape
        (num_points, 2), with one column per acquisition function, both to be minimized. This
        shape matches the objective values expected by NSGA-II in pymoo: one row per individual
        and one column per objective.

    Raises
    ------
    ValueError
        When called, if an acquisition function does not return one value per point of `x`.

    Notes
    -----
    Acquisition functions to be maximized must be negated, for example with `negate`.
    """

    acq_funcs = (acq_init_1(state), acq_init_2(state))

    def bi_acquisition(x: np.ndarray) -> np.ndarray:

        values = [np.asarray(acq(x), dtype=float).reshape(-1) for acq in acq_funcs]

        for val in values:
            if val.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Acquisition functions must return one value per point, but got "
                    f"{val.shape[0]} values for {x.shape[0]} points."
                )

        return np.column_stack(values)

    return bi_acquisition
