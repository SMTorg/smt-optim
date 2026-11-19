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
            return -np.asarray(acq(x), dtype=float)

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
        per point, of shape (num_points,) or (num_points, 1), to be minimized.
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
        When called, if an acquisition function does not return values of shape (num_points,)
        or (num_points, 1).

    Notes
    -----
    Acquisition functions to be maximized must be negated, for example with `negate`.
    """

    acq_funcs = (acq_init_1(state), acq_init_2(state))

    def bi_acquisition(x: np.ndarray) -> np.ndarray:

        n_points = x.shape[0]
        values = []

        for acq in acq_funcs:
            val = np.asarray(acq(x), dtype=float)

            # check the shape before flattening:
            if val.shape not in [(n_points,), (n_points, 1)]:
                raise ValueError(
                    f"Acquisition functions must return shape ({n_points},) or "
                    f"({n_points}, 1), but got {val.shape}."
                )

            values.append(val.reshape(-1))

        return np.column_stack(values)

    return bi_acquisition
