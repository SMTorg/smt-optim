from dataclasses import dataclass

import numpy as np
import scipy.optimize as so
import scipy.stats as stats

from smtoptim.utils.constraints import compute_rscv_sp

@dataclass
class MultistartResult:
    x: np.ndarray
    fun: float
    multi_x0: np.ndarray
    multi_x: np.ndarray
    multi_f: np.ndarray
    multi_rscv: np.ndarray
    multi_sp_res: list          # ScipPy results

def multistart_minimize(func, bounds, **kwargs):

    num_dim = bounds.shape[0]

    # constraints
    constraints = kwargs.pop('constraints', [])
    num_cstr = len(constraints)

    # multistart number
    n_start = kwargs.pop('n_start', 10*num_dim)

    multi_x0 = kwargs.pop('multi_x0', None)

    # tolerance
    tol = kwargs.pop('tol', np.sqrt(np.finfo(float).eps))

    # max iteration per start
    max_iter = kwargs.pop('max_iter', 50*num_dim)

    # seed for reproducibility
    seed = kwargs.pop('seed', None)

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    if multi_x0 is None:
        sampler = stats.qmc.LatinHypercube(d=num_dim, seed=seed)    # (tested) with random_state = None -> random
        multi_x0 = sampler.random(n_start)
        multi_x0 = stats.qmc.scale(multi_x0, bounds[:, 0], bounds[:, 1])
    else:
        n_start = multi_x0.shape[0]

    multi_x = np.empty_like(multi_x0)
    multi_f = np.empty(multi_x0.shape[0])
    multi_rscv = np.zeros(multi_x0.shape[0])
    multi_sp_res = []

    # unconstrained problem --> use L-BFGS-B
    if num_cstr == 0:
        method = "L-BFGS-B"

    # constrained problem --> use SLSQP
    else:
        method = "SLSQP"

    for i in range(multi_x0.shape[0]):

        res = so.minimize(func,
                          x0=multi_x0[i, :],
                          bounds=bounds,
                          method=method,
                          constraints=constraints,
                          tol=tol,
                          options={"maxiter": 50 * num_dim})

        # check bounds
        x = np.clip(res.x, bounds[:, 0], bounds[:, 1])
        multi_x[i, :] = x
        multi_f[i] = func(res.x)

        multi_sp_res.append(res)

        if num_cstr > 0:
            multi_rscv[i] = compute_rscv_sp(x, constraints)

    if num_cstr == 0:
        feas_mask = np.full(n_start, True)
    else:
        feas_mask = multi_rscv <= tol*2

    idx = np.argmin(multi_f[feas_mask])
    fmin = multi_f[feas_mask][idx]
    xmin = multi_x[feas_mask][idx]

    # add final optimization round

    res = MultistartResult(
        x = xmin,
        fun = fmin,
        multi_x0 = multi_x0,
        multi_x = multi_x,
        multi_f = multi_f,
        multi_rscv = multi_rscv,
        multi_sp_res = multi_sp_res,
    )

    return res










