from typing import overload

import numpy as np


# @overload
# def compute_rscv(cstr_array: np.ndarray, cstr_config: list, g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:
#
#     scv = np.full_like(cstr_array, 0.0)     # Square Constraint Violation
#
#     for c_id, c_config in enumerate(cstr_config):
#
#         if c_config.type in ["less", "greater"]:
#             valid_mask = cstr_array[:, c_id] <= g_tol
#             scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2
#
#         elif c_config.type == "equal":
#             valid_mask = np.abs(cstr_array[:, c_id]) <= h_tol
#             scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2
#
#         else:
#             raise Exception(f"{c_config.type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'.")
#
#     rscv = np.sqrt(scv.sum(axis=1))
#
#     return rscv


# @overload
# def compute_rscv(c: np.ndarray, c_type: list[str] | None, g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:
#
#     if c.shape[1] != len(c_type):
#         raise Exception("Number of constraint types must correspond to number of constraints.")
#
#     scv = np.full_like(c, 0.0)     # Square Constraint Violation
#
#     for c_id in range(c.shape[1]):
#
#         if c_type[c_id] == "less":
#             scv[:, c_id] = np.maximum(0, c[:, c_id])**2
#         elif c_type[c_id] == "greater":
#             scv[:, c_id] = np.minimum(0, c[:, c_id])**2
#         elif c_type[c_id] == "equal":
#             scv[:, c_id] = c[:, c_id]**2
#         else:
#             raise Exception(f"{c_type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'.")
#
#     rscv = np.sqrt(scv.sum(axis=1))
#
#     return rscv



def compute_rscv(cstr_values: np.ndarray, cstr_configs: list) -> np.ndarray:

    # cstr_values of shape (num_data, num_cstr)
    num_data = cstr_values.shape[0]

    rscv = np.zeros(num_data)

    for c_id, c_config in enumerate(cstr_configs):

        if c_config.equal is not None:
            unfeasible = cstr_values[:, c_id] != c_config.equal
            rscv[unfeasible] += (cstr_values[unfeasible, c_id] - c_config.equal)**2

        else:
            if c_config.lower is not None:
                unfeasible = cstr_values[:, c_id] < c_config.lower
                rscv[unfeasible] += (cstr_values[unfeasible, c_id] - c_config.lower)** 2

            if c_config.upper is not None:
                unfeasible = cstr_values[:, c_id] > c_config.upper
                rscv[unfeasible] += (cstr_values[unfeasible, c_id] - c_config.upper) ** 2

    return np.sqrt(rscv)

# TODO: move to multistart.py
def compute_rscv_sp(x: np.ndarray, cstr_list: list[dict]) -> float:
    """
    SciPy wrapper to compute the Root Squared Constraint Violation (RSCV).

    :param x:
    :param cstr_list:
    :return:
    """

    num_cstr = len(cstr_list)
    scv = np.empty(num_cstr)

    for c_id, c_dict in enumerate(cstr_list):

        c_type = c_dict["type"]
        if c_type == "ineq":
            scv[c_id] = min(0, c_dict["fun"](x))**2
        elif c_type == "eq":
            scv[c_id] = c_dict["fun"](x)**2

    return np.sqrt(np.sum(scv))



