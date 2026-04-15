from dataclasses import dataclass
import copy

import numpy as np
import scipy.optimize as so
import scipy.stats as stats
from smt.design_space import DesignSpace

from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
import smt.design_space as ds

from smt_optim.utils.constraints import compute_rscv_sp

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

    # SciPy optimization method
    method = kwargs.pop('method', None)

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

    if method is None:
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
        feas_mask = multi_rscv <= np.sqrt(tol) #*2
    if len(multi_f[feas_mask]) != 0:
        idx = np.argmin(multi_f[feas_mask])
        fmin = multi_f[feas_mask][idx]
        xmin = multi_x[feas_mask][idx]
    else:
        idx = np.argmin(multi_rscv)
        fmin = multi_f[idx]
        xmin = multi_x[idx]

    # TODO: add final optimization round

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




def mixvar_multistart_minimize(func, design_space: ds.DesignSpace, constraints: list = [], **kwargs):


    method = kwargs.get("method", "Cobyla")
    tol = kwargs.get("tol", np.sqrt(np.finfo(float).eps))

    seed = kwargs.get("seed", None)

    n_cont = 0
    scaled_dv = []
    cont_mask = np.full(design_space.n_dv, False)
    for idx, dv in enumerate(design_space.design_variables):
        if isinstance(dv, ds.FloatVariable):
            scaled_dv.append(ds.FloatVariable(0, 1))
            n_cont += 1
            cont_mask[idx] = True
        else:
            scaled_dv.append(dv)

    scaled_ds = DesignSpace(scaled_dv)
    cont_bounds = np.array([[0, 1]] * n_cont)

    # generate mixvar LHS
    # issue with seed parameter
    # sampler = MixedIntegerSamplingMethod(LHS,
    #                                      scaled_ds,
    #                                      criterion="ese",
    #                                      seed=seed)
    sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(), criterion="ese", seed=seed)

    n_small = kwargs.pop("n_start", 20)
    n_large = kwargs.pop("n_large", n_small * 10)

    x_large = sampler(n_large)
    # quick fix for the seed parameter issue
    x_large, _ = design_space.fold_x(x_large)
    func_val = np.empty(n_large)
    rscv = np.zeros(n_large)

    for idx in range(x_large.shape[0]):
        func_val[idx] = func(x_large[idx, :])

        for c_idx, c_dict in enumerate(constraints):
            if c_dict["type"] == "ineq":
                rscv[idx] += min(0, c_dict["fun"](x_large[idx, :]))**2
            else:
                rscv[idx] += c_dict["fun"](x_large[idx, :]) ** 2

    sorted_idx = np.lexsort((func_val, rscv))
    x_small = x_large[sorted_idx, :][:n_small, :]

    multi_x0 = copy.deepcopy(x_small)
    multi_x = x_small
    multi_f = func_val[sorted_idx][:n_small]
    multi_rscv = rscv[sorted_idx][:n_small]
    multi_sp_res = []

    if n_cont > 0 and method is not None:

        def wrapper(x_cont, x_ref, fun=func):
            x = x_ref
            x[cont_mask] = x_cont
            return fun(x)

        for idx in range(x_small.shape[0]):

            x_ref = x_small[idx, :]
            x0 = x_ref[cont_mask]

            multi_x0[idx, :] = x_ref

            wrapped_func = lambda x, o=func, f=wrapper: f(x, x_ref=x_ref, fun=o)

            wrapped_cstr = copy.deepcopy(constraints)
            for c_idx, c_dict in enumerate(wrapped_cstr):
                c_dict["fun"] = lambda x, c=constraints[c_idx]["fun"], f=wrapper: f(x, x_ref=x_ref, fun=c)

            res = so.minimize(wrapped_func, x0=x0, bounds=cont_bounds, constraints=wrapped_cstr, method=method, tol=tol)

            x_ref[cont_mask] = res.x

            multi_x[idx, :] = x_ref
            multi_f[idx] = res.fun

            for c_idx, c_dict in enumerate(constraints):
                # TODO: add tolerance for RSCV similar to tolerance given to SciPy solver
                if c_dict["type"] == "ineq":
                    multi_rscv[idx] += min(0, c_dict["fun"](x_large[idx, :]))**2
                else:
                    multi_rscv[idx] += c_dict["fun"](x_large[idx, :]) ** 2

            # if multi_rscv[idx] <= tol:
            #     multi_rscv[idx] = 0.

            multi_sp_res.append(res)


    best_idx = np.lexsort((multi_f, multi_rscv))[0]

    res = MultistartResult(
        x = multi_x[best_idx, :],
        fun = multi_f[best_idx],
        multi_x0 = multi_x0,
        multi_x = multi_x,
        multi_f = multi_f,
        multi_rscv = multi_rscv,
        multi_sp_res = multi_sp_res,
    )

    return res








