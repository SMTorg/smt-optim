import numpy as np
from scipy.spatial.distance import cdist

from moocore import hypervolume as moocore_hv
from pymoo.core.problem import Problem as PymooProblem


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
    """
    Return the non-dominated objective vectors from ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Array of shape ``(n_samples, n_objectives)`` containing objective
        values for each sample.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_pareto, n_objectives)`` containing the
        non-dominated objective vectors (the Pareto front).

    Notes
    -----
    Assumes a minimization problem for all objectives and no constraints.
    """

    pareto_mask = get_pareto_mask(Y)
    pareto = Y[pareto_mask]
    return pareto


def get_pf_from_dataset(dataset, ctol: float = 1e-4, fid: int = -1) -> np.ndarray:

    data_dict = dataset.export_as_dict()
    obj = data_dict["obj"]
    rscv = data_dict["rscv"]
    fidelity = data_dict["fidelity"]

    feas_mask = rscv <= ctol

    if fid == -1:
        fid = np.max(fidelity)

    fid_mask = fidelity[feas_mask] == fid

    pareto_front = get_pareto_front(obj[feas_mask][fid_mask])

    return pareto_front


def hypervolume_2d(pf: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the 2D hypervolume indicator  he hypervolume of the Pareto front.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, 2)
        Pareto front.

    ref: np.ndarray of shape (2, )
        Reference objective values.

    Returns
    -------
    float
        Hypervolume indicator value.

    Notes:
        Assume both objective are minimized.
    """

    if pf.shape[1] != 2 or ref.shape[0] != 2:
        raise Exception(
            "Current hypervolume implementation is only for bi-objective optimization."
        )

    sorted_idx = np.argsort(pf[:, 0])
    sorted_pf = pf[sorted_idx]

    hv = 0.0
    prev_f2 = ref[1]

    for idx in range(sorted_pf.shape[0]):
        f1 = sorted_pf[idx, 0]
        f2 = sorted_pf[idx, 1]

        width = ref[0] - f1
        height = prev_f2 - f2

        if width > 0 and height > 0:
            hv += width * height

        prev_f2 = min(prev_f2, f2)

    return hv


def hypervolume(pf: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the hypervolume indicator of the Pareto front.

    Uses the `moocore` implementation:
    https://multi-objective.github.io/moocore/python/reference/generated/moocore.hypervolume.html

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, 2)
        Pareto front.

    ref: np.ndarray of shape (2, )
        Reference objective values.

    Returns
    -------
    float
        Hypervolume indicator value.

    Notes:
        Assume both objective are minimized.
    """

    return moocore_hv(pf, ref=ref)


def spacing(pf: np.ndarray) -> float:
    """
    Compute the spacing indicator of the Pareto front (Schott, 1995). A lower value is better.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, num_objectives)
        Pareto front.

    Returns
    -------
    float
        Spacing indicator value.

    Notes:
        Assume both objective are minimized.
    """

    num_pf = pf.shape[0]

    if num_pf <= 1:
        return np.nan

    distances = cdist(pf, pf, "cityblock")
    np.fill_diagonal(distances, np.inf)

    d1 = np.min(distances, axis=1)
    d1_mean = np.mean(d1)

    value = np.sqrt(1 / (num_pf - 1) * np.sum((d1_mean - d1) ** 2))

    return value


class PymooStateWrapper(PymooProblem):
    def __init__(self, state):

        self.state = state

        self.state.scale_dataset(False)
        self.state.build_models()

        prob = self.state.problem

        l_bounds = []
        u_bounds = []
        for idx, var in enumerate(prob.design_space.design_variables):
            l_bounds.append(var.lower)
            u_bounds.append(var.upper)

        self.l_bounds = np.array(l_bounds)
        self.u_bounds = np.array(u_bounds)

        self.f_callables = []
        self.g_callables = []
        self.h_callables = []

        for o_id, o_config in enumerate(prob.obj_configs):
            self.f_callables.append(
                lambda x, f=self.state.obj_models[o_id].predict_values: f(x).ravel()
            )

        for c_id, c_config in enumerate(prob.cstr_configs):
            if c_config.equal is not None:
                self.h_callables.append(
                    lambda x, f=self.state.cstr_models[c_id].predict_values, val=c_config.equal: (
                        f(x).ravel() - val
                    )
                )

            else:
                if c_config.lower is not None:
                    self.g_callables.append(
                        lambda x, f=self.state.cstr_models[c_id].predict_values, val=c_config.lower: (
                            val - f(x).ravel()
                        )
                    )

                if c_config.upper is not None:
                    self.g_callables.append(
                        lambda x, f=self.state.cstr_models[c_id].predict_values, val=c_config.upper: (
                            f(x).ravel() - val
                        )
                    )

        super().__init__(
            n_var=prob.num_dim,
            n_obj=prob.num_obj,
            n_eq_constr=len(self.h_callables),
            n_ieq_constr=len(self.g_callables),
            xl=self.l_bounds,
            xu=self.u_bounds,
        )

    def _evaluate(self, x, out, *args, **kwargs):

        num_pt = x.shape[0]

        x_scaled = (x - self.l_bounds) / (self.u_bounds - self.l_bounds)

        out["F"] = np.full((num_pt, self.n_obj), np.nan)

        if self.n_eq_constr > 0:
            out["H"] = np.empty((num_pt, self.n_eq_constr))

        if self.n_ieq_constr > 0:
            out["G"] = np.empty((num_pt, self.n_ieq_constr))

        for o_idx in range(self.n_obj):
            out["F"][:, o_idx] = self.f_callables[o_idx](x_scaled).ravel()

        for h_idx in range(self.n_eq_constr):
            out["H"][:, h_idx] = self.h_callables[h_idx](x_scaled).ravel()

        for g_idx in range(self.n_ieq_constr):
            out["G"][:, g_idx] = self.g_callables[g_idx](x_scaled).ravel()


def purity(pf: np.ndarray, ref_pf: np.ndarray) -> float:
    """
    Compute the purity indicator of the Pareto front.
    Purity is the ratio of points in the obtained Pareto front that are not
    strictly dominated by any point in the reference Pareto front.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, num_objectives)
        Pareto front approximation.
    ref_pf: np.ndarray of shape (num_ref_points, num_objectives)
        Reference Pareto front.

    Returns
    -------
    float
        Purity indicator value (between 0.0 and 1.0).
    """
    if len(pf) == 0:
        return 0.0
    dominated = 0
    for p in pf:
        # A point is dominated if any point in ref_pf is <= in all dims and < in at least one
        is_dom = np.any(np.all(ref_pf <= p, axis=1) & np.any(ref_pf < p, axis=1))
        if is_dom:
            dominated += 1
    return 1.0 - dominated / len(pf)


def gamma_spread(pf: np.ndarray) -> float:
    """
    Compute the Gamma-spread (Maximum Spread) indicator of the Pareto front.
    It measures the maximum Euclidean distance between any two points in the
    Pareto front.

    Parameters
    ----------
    pf: np.ndarray of shape (num_points, num_objectives)
        Pareto front.

    Returns
    -------
    float
        Maximum spread indicator value.
    """
    if len(pf) <= 1:
        return np.nan

    from scipy.spatial.distance import pdist

    distances = pdist(pf)
    if len(distances) == 0:
        return np.nan
    return float(np.max(distances))
