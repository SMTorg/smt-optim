import numpy as np

from smt_optim.utils.constraints import compute_rscv


def get_fmin(
    f: np.ndarray,
    c: np.ndarray | None = None,
    c_type: list[str] | None = None,
    rscv_tol: float = 0.0,
) -> float:

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
