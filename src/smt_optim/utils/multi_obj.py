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


def get_pf_from_dataset(
    dataset, ctol: float = 1e-4, fid: int = -1, return_dict: bool = False
) -> np.ndarray | dict:

    data_dict = dataset.export_as_dict()
    obj = data_dict["obj"]
    rscv = data_dict["rscv"]
    fidelity = data_dict["fidelity"]

    feas_mask = rscv <= ctol

    if fid == -1:
        fid = np.max(fidelity)

    fid_mask = fidelity == fid

    fid_feas_mask = fid_mask & feas_mask

    pareto_mask = get_pareto_mask(obj[fid_feas_mask])

    pareto_front = obj[fid_feas_mask][pareto_mask]

    if return_dict:
        data = {
            "x": data_dict["x"][fid_feas_mask][pareto_mask],
            "obj": pareto_front,
            "cstr": data_dict["cstr"][fid_feas_mask][pareto_mask],
            "rscv": rscv[fid_feas_mask][pareto_mask],
            "fidelity": fidelity[fid_feas_mask][pareto_mask],
        }

        return data

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
    def __init__(self, state, scaled: bool = False, train: bool = True):
        """
        Create a Pymoo problem instance for use with Pymoo algorithms, such as NSGA-II for multi-objective optimization.

        The quantities of interest are modeled using the surrogate models.

        Parameters
        ----------
        state : State
        scaled : bool
            Normalize the input variables and standardize the quantities of interest
            (before training the surrogate models, if applicable).
        train : bool
            Train the surrogate models.

        Notes
        -----
        When using `PymooStateWrapper` after a Bayesian optimization (BO) iteration,
        `train` should be set to `True`, as the surrogate models are not trained on
        the most recently evaluated infill point.
        """

        self.state = state
        self.scaled = scaled
        self.train = train

        if not self.state.problem.design_space.is_all_cont:
            raise ValueError(
                "PymooStateWrapper currently requires a continuous optimization problem."
            )

        if self.train:
            self.state.scale_dataset(self.scaled)
            self.state.build_models()

        prob = self.state.problem

        if not self.scaled:
            l_bounds = []
            u_bounds = []

            for idx, var in enumerate(prob.design_space.design_variables):
                l_bounds.append(var.lower)
                u_bounds.append(var.upper)

            self.l_bounds = np.array(l_bounds)
            self.u_bounds = np.array(u_bounds)

        else:
            self.l_bounds = np.zeros(prob.num_dim)
            self.u_bounds = np.ones(prob.num_dim)

        self.f_callables = []
        self.g_callables = []
        self.num_g = 0
        self.h_callables = []
        self.num_h = 0

        ineq_bounds = []
        self.num_g_lower = 0
        self.num_g_upper = 0

        for o_id, o_config in enumerate(prob.obj_configs):
            self.f_callables.append(
                lambda x, f=self.state.obj_models[o_id].predict_values: f(x).ravel()
            )

        for c_id, c_config in enumerate(prob.cstr_configs):
            if c_config.equal is not None:
                self.h_callables.append(
                    lambda x, f=self.state.cstr_models[c_id].predict_values, val=c_config.equal: (
                        f(x).ravel() - self.state.cstr_equal[c_id]
                    )
                )
                self.num_h += 1

            else:
                self.g_callables.append(
                    lambda x, f=self.state.cstr_models[c_id].predict_values: f(
                        x
                    ).ravel()
                )

                ineq_bounds.append(np.full(2, np.nan))

                if c_config.lower is not None:
                    ineq_bounds[-1][0] = self.state.cstr_lower[c_id]
                    self.num_g += 1

                if c_config.upper is not None:
                    ineq_bounds[-1][1] = self.state.cstr_upper[c_id]
                    self.num_g += 1

        if len(self.g_callables):
            self.ineq_bounds = np.array(ineq_bounds)

            self.g_lower_mask = np.where(np.isnan(self.ineq_bounds[:, 0]), False, True)
            self.g_upper_mask = np.where(np.isnan(self.ineq_bounds[:, 1]), False, True)

        super().__init__(
            n_var=prob.num_dim,
            n_obj=prob.num_obj,
            n_eq_constr=self.num_h,
            n_ieq_constr=self.num_g,
            xl=self.l_bounds,
            xu=self.u_bounds,
        )

    def _evaluate(self, x, out, *args, **kwargs):

        num_pt = x.shape[0]

        x_scaled = (x - self.l_bounds) / (self.u_bounds - self.l_bounds)

        # sample objectives
        out["F"] = np.full((num_pt, self.n_obj), np.nan)
        for o_idx in range(self.n_obj):
            out["F"][:, o_idx] = self.f_callables[o_idx](x_scaled).ravel()

        # sample equality constraints
        if self.n_eq_constr > 0:
            out["H"] = np.empty((num_pt, self.n_eq_constr))
            for h_idx in range(self.n_eq_constr):
                out["H"][:, h_idx] = self.h_callables[h_idx](x_scaled).ravel()

        if self.n_ieq_constr > 0:
            g_lower = np.empty((num_pt, self.ineq_bounds.shape[0]))
            g_upper = np.empty((num_pt, self.ineq_bounds.shape[0]))

            # sample inequality constraints
            if self.n_ieq_constr > 0:
                for g_idx, g_call in enumerate(self.g_callables):
                    g_vals = self.g_callables[g_idx](x_scaled).ravel()

                    if not np.isnan(self.ineq_bounds[g_idx, 0]):
                        g_lower[:, g_idx] = self.ineq_bounds[g_idx, 0] - g_vals

                    elif not np.isnan(self.ineq_bounds[g_idx, 1]):
                        g_upper[:, g_idx] = g_vals - self.ineq_bounds[g_idx, 1]

                out["G"] = np.hstack(
                    (g_lower[:, self.g_lower_mask], g_upper[:, self.g_upper_mask])
                )
