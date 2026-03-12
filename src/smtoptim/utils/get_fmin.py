

import numpy as np

from smtoptim.utils.constraints import compute_rscv


def get_fmin(f: np.ndarray, c: np.ndarray | None = None, c_type: list[str] | None = None, rscv_tol: float = 0.0) -> float:

    if c is None:
        return min(f)

    else:
        if c_type is None:
            c_type = ["less" for _ in range(c.shape[1])]

        rscv = compute_rscv(c, c_type)

        feasible_mask = np.where(rscv <= rscv_tol, True, False)

        if np.any(feasible_mask):
            return f[feasible_mask].min()
        else:
            idx = np.argmin(rscv)
            return f[idx]

    # no constraint
    if optimizer.num_cstr == 0 or fmin_crit == "fmin":
        idx = optimizer.yt_scaled[-1].argmin()

    elif fmin_crit == "min_rscv":
        ct_rscv = self.compute_rscv(optimizer.ct_scaled[-1], optimizer.cstr_config)
        feas_mask = np.where(ct_rscv <= rscv_tol, True, False)

        if np.any(feas_mask):
            idx = np.argmin(np.where(feas_mask, optimizer.yt_scaled[-1][:, 0], np.inf))
        else:
            idx = np.argmin(ct_rscv)

    elif fmin_crit == "mean_rscv":
        rscv = self.compute_rscv(optimizer.yt_scaled[-1], optimizer.cstr_config)
        mean_rscv = rscv.mean()

        feas_mask = np.where(rscv <= mean_rscv, True, False)
        idx = np.argmin(np.where(feas_mask, optimizer.yt_scaled[-1][:, 0], np.inf))

    else:
        raise Exception(f"{fmin_crit} is not a valid fmin_crit")

    fmin = optimizer.yt_scaled[-1][idx, 0]

    return fmin

# def get_fmin_from_dataset(state, ctol):
#
#     if state.problem.num_obj > 1:
#         raise Exception("Not yet implemented for multi-objective dataset.")
#
#     indices = np.arange(state.problem.num_obj+state.problem.num_cstr, 1)
#
#     state.dataset.export_data(indices, state.dataset.fidelities[-1])
#
#
#
#     pass

