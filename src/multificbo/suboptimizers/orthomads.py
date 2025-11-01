import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass


""" 
<--- WORK IN PROGRESS --->
"""


@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: np.ndarray
    success: bool = False
    num_iter: int = 0
    num_eval: int = 0
    cstr: np.ndarray = None
    x_history: list = None
    fun_history: list = None
    cstr_history: list = None
    DD_history: list[np.ndarray] = None


def extreme_barrier(x: np.ndarray, func: callable, bounds: np.ndarray = None, constraints: list[callable] = None):

    f_value = np.full(x.shape[0], np.inf)

    if bounds is not None:
        mask = np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1]), axis=1)
    else:
        mask = np.ones(x.shape[0], dtype=bool)

    if constraints is not None:
        for g in constraints:
            mask &= (g(x) <= 0).all(axis=-1) if g(x).ndim > 1 else (g(x) <= 0)

    if np.any(mask):
        f_value[mask] = func(x[mask])
    else:
        pass

    return f_value


def orthomads(func: callable,
              x0: np.ndarray,
              max_iter: int = 100,
              bounds: np.ndarray = None,
              constraints: list = None,
              verbose: bool = False,
              history: bool = False,
              Delta_min: float = 1e-14,
              Delta0: float = 1.0,
              tau: float = 0.5,
              successive_max: int = 20,
              debug: bool = False):

    Delta = Delta0
    k = 0

    f0 = extreme_barrier(x0.reshape(1, -1), func, bounds, constraints).item()
    f = f0

    # eps = np.finfo(float).eps
    eps = 2.3e-16
    # eps = 5e-16

    dim = len(x0)

    x = x0.reshape(dim, 1)
    H = np.empty((dim, dim))
    B = np.empty((dim, dim))
    DD = np.empty((dim, 2 * dim))
    xx = np.empty((dim, 2 * dim))
    fxx = np.empty(2 * dim)
    L_inf = np.empty(dim)
    eye = np.eye(dim)

    halton = qmc.Halton(d=dim)

    if history:
        x_history = [x0]
        fun_history = [f]

        if debug:
            DD_history = []

    successive_counter = 0
    continue_opt = True

    while continue_opt:

        k += 1

        # 1. parameter update step
        delta = min(Delta, (Delta**2)/Delta0)   # set maximum step size
        delta = max(delta, eps)

        # 2. search step

        # 3. poll step
        v = halton.random(1).T
        u = v / np.sqrt(v*v).sum()

        np.copyto(H, eye)
        H -= 2 * (u @ u.T)

        np.max(np.abs(H), axis=0, out=L_inf)

        scale = Delta / delta
        B[:] = np.round(scale * H / L_inf)

        DD[:, :dim], DD[:, dim:] = B, -B

        if history and debug:
            DD_history.append(delta*DD)

        xx[:] = x + delta * DD
        fxx[:] = extreme_barrier(xx.T, func, bounds, constraints)

        idx_min = fxx.argmin()
        if fxx[idx_min] < f:
            f = fxx[idx_min]
            x[:] = xx[:, [idx_min]]
            Delta = min(Delta0, Delta / tau)    # increase step size
            successive_counter = 0
        else:
            Delta *= tau                        # decrease step size
            Delta = max(Delta_min, Delta)

            if Delta <= Delta_min:
                successive_counter += 1

        # verify stopping criteria
        if k > max_iter:
            continue_opt = False  # stop optimization if max num of iter is reached
            stop_message = "Max number of iteration reached"

        if Delta <= Delta_min and successive_counter > successive_max:
            continue_opt = False  # stop optimization if step size is smaller than criterion
            stop_message = "Delta_min and successive_counter reached"

        if history:
            x_history.append(x.ravel())
            fun_history.append(f)

        if verbose:
            print(f"k={k} | x={x} | fun={f} | Delta={Delta} | delta={delta}")


    res = OptimizationResult(
        x.reshape(-1),
        f
    )

    res.num_iter = k
    res.num_eval = 0

    if f < f0:
        res.success = True
    else:
        res.success = False

    res.stop_message = stop_message

    if history:
        res.x_history = x_history
        res.fun_history = fun_history

        if debug:
            res.DD_history = DD_history

    # return OptimizationResult dataclass
    return res