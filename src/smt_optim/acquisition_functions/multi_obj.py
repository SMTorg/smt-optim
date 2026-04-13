from functools import partial

import numpy as np

import scipy.stats as stats
from scipy.special import erfcx



def bi_obj_pi(x_pred: np.ndarray, obj_vals: np.ndarray, models: list):

    s0 = np.sqrt(models[0].predict_variances(x_pred)).item()
    s1 = np.sqrt(models[1].predict_variances(x_pred)).item()

    if s0 <= 0. or s1 <= 0.:
        return 0.

    y0 = models[0].predict_values(x_pred).item()
    y1 = models[1].predict_values(x_pred).item()

    cdf_vals = np.empty((obj_vals.shape[0], 2))
    cdf_vals[:, 0] = stats.norm.cdf((obj_vals[:, 0] - y0)/s0)
    cdf_vals[:, 1] = stats.norm.cdf((obj_vals[:, 1] - y1)/s1)

    pi = cdf_vals[0, 0]
    pi += np.sum((cdf_vals[1:, 0] - cdf_vals[:-1, 0]) * cdf_vals[1:, 1])
    pi += (1. - cdf_vals[-1, 0]) * cdf_vals[-1, 1]

    return pi


def init_bi_obj_pi(state):

    obj = state.scaled_dataset.export_as_dict()["obj"]
    pareto = get_pareto_front(obj)
    # first objective needs to be sorted? what about the second to compute Y1?
    sorted_idx = np.argsort(pareto[:, 0])
    sorted_pareto = pareto[sorted_idx, :]

    models = state.obj_models

    bi_obj_pi_func = partial(bi_obj_pi, obj_vals=sorted_pareto, models=models)

    return bi_obj_pi_func



def bi_obj_ei(x_pred: np.ndarray, obj_vals: np.ndarray, models: list):
    # TODO: debugging required
    s = np.array([
        np.sqrt(models[0].predict_variances(x_pred)).item(),
        np.sqrt(models[1].predict_variances(x_pred)).item()
    ])

    if np.any(s <= 1.e-15):
        return 0.

    y = np.array([
        models[0].predict_values(x_pred).item(),
        models[1].predict_values(x_pred).item(),
    ])

    cdf = np.empty((obj_vals.shape[0], 2))
    pdf = np.empty((obj_vals.shape[0], 2))
    for idx in range(2):
        z = (obj_vals[:, idx] - y[idx]) / s[idx]
        z = np.clip(z, -10, 10)
        cdf[:, idx] = stats.norm.cdf(z)
        pdf[:, idx] = stats.norm.pdf(z)

    # TODO: avoid redundant computation
    PI = bi_obj_pi(x_pred, obj_vals, models)

    if PI > 1e-15:
        Y = np.zeros(2)

        for i in range(2):
            j = 1 if i == 0 else 0

            Y[i] = y[i] * cdf[0, i] - s[i] * pdf[0, i]
            Y[i] += np.sum((y[i] * cdf[1:, i] - s[i] * pdf[1:, i] - (y[i] * cdf[:-1, i] - s[i] * pdf[:-1, i])) * cdf[1:, j])
            Y[i] += (y[i] * cdf[-1, i] - s[i] * pdf[-1, i]) * cdf[-1, j]
            Y[i] /= PI

        EI = PI * np.linalg.norm(Y - y)
    else:
        EI = 0

    return EI


def init_bi_obj_ei(state):

    obj = state.scaled_dataset.export_as_dict()["obj"]
    pareto = get_pareto_front(obj)

    # first objective needs to be sorted?
    sorted_idx = np.argsort(pareto[:, 0])
    sorted_pareto = pareto[sorted_idx, :]

    models = state.obj_models

    bi_obj_ei_func = partial(bi_obj_ei, obj_vals=sorted_pareto, models=models)

    return bi_obj_ei_func



def get_pareto_mask(Y: np.ndarray) -> np.ndarray:

    n = Y.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        # A point is dominated if another point is <= in all objectives
        # and strictly < in at least one
        dominates = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)

        # If any point dominates i -> i is not Pareto
        if np.any(dominates):
            is_pareto[i] = False

    return is_pareto


def get_pareto_front(Y: np.ndarray) -> np.ndarray:
    pareto_mask = get_pareto_mask(Y)
    pareto = Y[pareto_mask]
    return pareto

