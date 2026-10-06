r"""
Post-processing tools to compare optimization algorithms on benchmark results.

The functions of this module operate on the nested dictionary returned by
:func:`~smt_optim.postprocessing.load_data.load_data`: ``data[algo][problem][instance]`` is a
:class:`pandas.DataFrame` holding the DOE of one optimization run (one ``doe.csv`` file).

Each DataFrame must contain the columns ``iter``, ``budget``, ``fidelity``, ``f0`` (objective value) and
``rscv`` (relative sum of constraint violations). Rows with ``iter == 0`` form the initial DOE. Additional
columns (for example ``time``) can be used to define custom cost metrics.

Three kinds of profiles are available:

- **Convergence profile** (:func:`get_convergence_profile`): best feasible objective value versus cost.
- **Data profile** (:func:`get_data_profile`): fraction of problem instances solved as a function of the cost [1]_.
- **Performance profile** (:func:`get_perf_profile`): fraction of problem instances solved within a given
  factor of the cost of the best algorithm [2]_.

A run solves an instance when a feasible point (``rscv <= ctol``) evaluated at the highest fidelity level
satisfies

.. math::
    f(\boldsymbol{x}) \leq f_\star + \tau (f_0 - f_\star),

where :math:`f_0` and :math:`f_\star` are the reference initial and best objective values of the problem,
shared by all algorithms (see :func:`compute_f0_fstar`), and :math:`\tau \in [0, 1]` is the tolerance.

The data and performance profiles are computed in two steps: :func:`get_solve_table` records, for every
(algorithm, problem, instance), the cost at which the instance was solved; the table is then compacted by
:func:`get_data_profile_from_table` or :func:`get_perf_profile_from_table`. :func:`get_data_profile` and
:func:`get_perf_profile` chain both steps.

References
----------
.. [1] Moré, J. J., & Wild, S. M. (2009). Benchmarking derivative-free optimization algorithms.
       SIAM Journal on Optimization, 20(1), 172-191.
.. [2] Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software with performance profiles.
       Mathematical Programming, 91(2), 201-213.
"""

import os
import pickle
import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd



def order_f0(df: pd.DataFrame, metric_name: str = "budget", ctol: float = 1e-4) -> pd.DataFrame:
    """
    Best feasible objective value found so far along a single run.

    Only feasible points evaluated at the highest fidelity level are considered. The returned history starts
    at the end of the initial DOE, whose value is the best of the whole initial DOE.

    Parameters
    ----------
    df : pandas.DataFrame
        Results of a single instance. Must contain the columns ``iter``, ``fidelity``, ``rscv``, ``f0`` and
        ``metric_name``.
    metric_name : str, default="budget"
        Column of `df` used as cost.
    ctol : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= ctol``.

    Returns
    -------
    pandas.DataFrame
        Columns ``iter``, `metric_name` and ``fmin``. ``fmin`` is the cumulative minimum of the feasible
        objective values, and is ``inf`` until a feasible point is found.
    """

    fidelity = df["fidelity"].to_numpy()
    fid_mask = fidelity == np.max(fidelity)

    feas_mask = df["rscv"].to_numpy() <= ctol

    fid_feas_mask = fid_mask & feas_mask

    iter = df["iter"].to_numpy()
    metric = df[metric_name].to_numpy()
    f0 = df["f0"].to_numpy()

    f0_inf = np.where(fid_feas_mask, f0, np.inf)
    f0_ordered = np.minimum.accumulate(f0_inf)

    iter0_mask = iter == 0
    iter0_fmin = np.min(f0_ordered[iter0_mask])

    iter0_max_id = np.where(iter0_mask)[0][-1]

    iter = iter[iter0_max_id:]
    metric = metric[iter0_max_id:]
    f0_ordered = f0_ordered[iter0_max_id:]
    f0_ordered[0] = iter0_fmin

    new_df = pd.DataFrame(data={
        "iter": iter,
        metric_name: metric,
        "fmin": f0_ordered,
    })

    return new_df


def get_convergence_profile(data: pd.DataFrame, metric_name: str = "budget", ctol: float = 1e-4) -> dict:
    """
    Best feasible objective value versus cost, for every algorithm and problem.

    For each problem, the cost values of all instances are merged on a common grid, and the best feasible
    objective value found so far (see :func:`order_f0`) is evaluated on it for each instance.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance] -> pandas.DataFrame``, as returned by
        :func:`~smt_optim.postprocessing.load_data.load_data`.
    metric_name : str, default="budget"
        Column of the instance DataFrames used as cost (x-axis).
    ctol : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= ctol``.

    Returns
    -------
    dict
        ``conv_data[algo][problem]`` is a dictionary with the keys:

        - `metric_name` : ndarray of shape (n_cost,), sorted cost values shared by all instances.
        - ``"max_metric"`` : ndarray of shape (n_instance,), final cost of each instance.
        - ``"f0"`` : ndarray of shape (n_cost, n_instance), best feasible objective of each instance.
        - ``"mean"``, ``"std"``, ``"min"``, ``"max"`` : ndarray of shape (n_cost,), statistics over the instances.

    Notes
    -----
    Instances that have not found a feasible point yet have an objective value of ``inf``. The statistics are
    therefore infinite (or NaN for ``std``) until all instances have found a feasible point.

    Examples
    --------
    >>> data, _ = load_data("data/unconstrained")
    >>> conv = get_convergence_profile(data, metric_name="iter")
    >>> conv["expected_improvement"]["Sphere_2d"]["mean"]
    """

    conv_data = {}


    for algo, a_data in data.items():

        if conv_data.get(algo, None) is None:
            conv_data[algo] = dict()

        for problem, p_data in a_data.items():

            num_inst = len(p_data.keys())

            max_metric = np.empty(num_inst)  # [i_data[-1] for i, i_data in enumerate(a_data.values())]

            all_metric = np.concatenate(
                [i_data[metric_name].to_numpy() for i_data in p_data.values()]
            )

            all_metric = np.unique(all_metric)
            all_metric = np.sort(all_metric)

            # fetch the metric final value for each instance
            for i, i_data in enumerate(p_data.values()):
                max_metric[i] = i_data[metric_name].to_numpy()[-1]

            all_f0 = np.full((all_metric.shape[0], num_inst), np.inf)

            for j, (instance, i_data) in enumerate(p_data.items()):

                ordered_data = order_f0(i_data, metric_name, ctol)

                idx = np.searchsorted(all_metric, ordered_data[metric_name])

                all_f0[idx, j] = ordered_data["fmin"]

                all_f0[:, j] = np.minimum.accumulate(all_f0[:, j], axis=0)

            conv_data[algo][problem] = {
                metric_name: all_metric,
                "max_metric": max_metric,
                "f0": all_f0,
                "mean": np.mean(all_f0, axis=1),
                "std": np.std(all_f0, axis=1),
                "min": np.min(all_f0, axis=1),
                "max": np.max(all_f0, axis=1),
            }

    return conv_data


F0_METHODS = ("max_feasible", "init_min", "init_max")


def compute_f0_fstar(data: dict, eps: float = 1e-4, method: str | None = None):
    """
    Reference initial value ``f0`` and best value ``fstar`` of each problem.

    The values are shared by all algorithms and instances of a problem, and are computed from feasible points
    evaluated at the highest fidelity level.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance] -> pandas.DataFrame``.
    eps : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= eps``.
    method : {"max_feasible", "init_min", "init_max"}, optional
        How the reference initial value ``f0`` is chosen. The default (None) is ``"max_feasible"``.

        - ``"max_feasible"``: largest feasible objective value over the whole run.
        - ``"init_min"``: smallest feasible objective value in the initial DOE (``iter == 0``).
        - ``"init_max"``: largest feasible objective value in the initial DOE (``iter == 0``).

    Returns
    -------
    dict
        ``f0_fstar[problem]`` is a dictionary with the keys ``"f0"`` (largest reference initial value over the
        instances and algorithms) and ``"fstar"`` (best feasible objective value over the instances and algorithms).

    Raises
    ------
    ValueError
        If `method` is unknown, or if `method` is ``"init_min"`` or ``"init_max"`` and an instance has no
        feasible point in its initial DOE.

    Notes
    -----
    A problem for which no feasible point was found keeps ``f0 = -inf`` and ``fstar = inf``.
    """

    if method is None:
        method = "max_feasible"

    if method not in F0_METHODS:
        raise ValueError(f"Unknown f0 method '{method}'. Choose from {F0_METHODS}.")

    f0_fstar = dict()

    for algo, a_data in data.items():
        for prob, p_data in a_data.items():

            if f0_fstar.get(prob, None) is None:
                f0_fstar[prob] = {
                    "fstar": np.inf,
                    "f0": -np.inf
                }

            for inst, i_data in p_data.items():
                # i_data in a pd.DataFrame
                max_fid = i_data["fidelity"].to_numpy().max()
                fid_feas_mask = (i_data["fidelity"].to_numpy() == max_fid) & (i_data["rscv"].to_numpy() <= eps)
                
                if np.any(fid_feas_mask):
                    all_f0 = i_data["f0"].to_numpy()
                    obj_values = all_f0[fid_feas_mask]
                    fstar = np.min(obj_values)

                    if method == "max_feasible":
                        f0 = np.max(obj_values)
                    else:
                        init_mask = fid_feas_mask & (i_data["iter"].to_numpy() == 0)
                        if not np.any(init_mask):
                            raise ValueError(
                                f"Method '{method}' requires a feasible point in the initial DOE, "
                                f"but none was found for {algo}/{prob}/{inst}."
                            )
                        init_values = all_f0[init_mask]
                        f0 = np.min(init_values) if method == "init_min" else np.max(init_values)

                    if fstar < f0_fstar[prob]["fstar"]:
                        f0_fstar[prob]["fstar"] = fstar

                    if f0 > f0_fstar[prob]["f0"]:
                        f0_fstar[prob]["f0"] = f0
                else:
                    continue

    return f0_fstar



def compute_threshold(f0_fstar: dict, tau: float = 0.1) -> dict:
    r"""
    Objective value an algorithm must reach to solve a problem.

    Parameters
    ----------
    f0_fstar : dict
        Reference values per problem, as returned by :func:`compute_f0_fstar`.
    tau : float, default=0.1
        Tolerance in [0, 1]. Smaller values are more demanding: ``tau = 0`` requires reaching ``fstar``.

    Returns
    -------
    dict
        ``thresholds[problem]`` is :math:`f_\star + \tau (f_0 - f_\star)`.
    """

    thresholds = dict()

    for prob, p_data in f0_fstar.items():
        thresholds[prob] = p_data["fstar"] + tau * (p_data["f0"] - p_data["fstar"])

    return thresholds


_X_COL = re.compile(r"^x\d+$")

_RESERVED = {"algo", "problem", "instance", "n_dim", "solved", "fmin"}


def _column_metric(name: str) -> Callable:
    return lambda df: df[name].to_numpy(dtype=float)


BUILTIN_METRICS = {
    "idx": lambda df: np.arange(len(df), dtype=float),
    "iter": _column_metric("iter"),
    "budget": _column_metric("budget"),
}


def _resolve_metrics(metrics: Sequence[str] | Mapping[str, Callable] | None) -> dict[str, Callable]:
    """Map metric names to callables `df -> array`. Names are built-ins, instance DataFrame columns, or mapping keys."""

    if metrics is None:
        metrics = list(BUILTIN_METRICS)

    if isinstance(metrics, Mapping):
        resolved = dict(metrics)
    else:
        resolved = {m: BUILTIN_METRICS.get(m, _column_metric(m)) for m in metrics}

    for name in resolved:
        if name in _RESERVED or name.startswith("max_"):
            raise ValueError(f"Metric name '{name}' is reserved.")

    return resolved


def _eval_metrics(metrics: dict[str, Callable], i_data: pd.DataFrame, where: str) -> dict[str, np.ndarray]:
    values = {}
    for name, fn in metrics.items():
        try:
            v = np.asarray(fn(i_data), dtype=float)
        except KeyError as e:
            raise ValueError(f"Metric '{name}': column {e} missing for {where}.") from e

        if v.shape != (len(i_data),) or np.isnan(v).any() or np.any(np.diff(v) < 0):
            raise ValueError(
                f"Metric '{name}' must be a NaN-free, non-decreasing (cumulative) cost with one value per row ({where})."
            )
        values[name] = v

    return values


def get_solve_table(
    data: dict,
    tau: float = 0.1,
    ctol: float = 1e-4,
    f0_method: str | None = None,
    metrics: Sequence[str] | Mapping[str, Callable] | None = None,
) -> pd.DataFrame:
    """
    Tabulate the cost at which each run solves its problem instance.

    This is the common first step of the data and performance profiles. A run solves an instance at the first
    row where a feasible point of the highest fidelity level reaches the threshold of the problem (see
    :func:`compute_threshold`); its cost is the value of each metric at that row.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance] -> pandas.DataFrame``, as returned by
        :func:`~smt_optim.postprocessing.load_data.load_data`.
    tau : float, default=0.1
        Tolerance in [0, 1] defining the threshold to reach (see :func:`compute_threshold`).
    ctol : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= ctol``.
    f0_method : {"max_feasible", "init_min", "init_max"}, optional
        Reference initial value used to define the threshold (see :func:`compute_f0_fstar`).
    metrics : sequence of str or mapping of str to callable, optional
        Cost metrics to tabulate. The default (None) is ``["idx", "iter", "budget"]``.

        - A sequence of names. A name is a built-in metric (``"idx"``, the row position; ``"iter"``;
          ``"budget"``) or any column of the instance DataFrames.
        - A mapping ``{name: function}``. Each ``function`` takes the instance DataFrame and returns one cost
          per row, e.g. ``{"cum_time": lambda df: df["time"].cumsum().to_numpy()}``.

        A metric must be a cumulative cost: NaN-free and non-decreasing along the rows.

    Returns
    -------
    pandas.DataFrame
        One row per (algorithm, problem, instance) with the columns:

        - ``algo``, ``problem``, ``instance`` : identifiers.
        - ``n_dim`` : number of design variables (columns ``x0``, ``x1``, ...).
        - ``solved`` : True if the threshold was reached.
        - ``fmin`` : best feasible objective value of the run, ``inf`` if none.
        - one column per metric : cost at which the instance was solved, ``inf`` if unsolved.
        - one ``max_<metric>`` column per metric : cost at the end of the run.

    Raises
    ------
    ValueError
        If a metric name is reserved (``algo``, ``problem``, ``instance``, ``n_dim``, ``solved``, ``fmin`` or
        starting with ``max_``), if a metric column is missing, or if a metric is not a NaN-free, non-decreasing
        cost with one value per row.

    See Also
    --------
    get_data_profile_from_table, get_perf_profile_from_table

    Notes
    -----
    The table can be reused to compute several profiles without recomputing the thresholds.

    Examples
    --------
    >>> data, _ = load_data("data/multifidelity")
    >>> table = get_solve_table(data, tau=0.1, ctol=1e-4, metrics=["budget", "iter"])
    >>> table.groupby("algo")["solved"].mean()
    """

    metrics = _resolve_metrics(metrics)
    thresholds = compute_threshold(compute_f0_fstar(data, ctol, f0_method), tau)

    rows = []
    for algo, a_data in data.items():
        for prob, p_data in a_data.items():
            for inst, i_data in p_data.items():

                fid = i_data["fidelity"].to_numpy()
                feas = (fid == fid.max()) & (i_data["rscv"].to_numpy() <= ctol)
                obj = np.where(feas, i_data["f0"].to_numpy(), np.inf)

                reached = obj <= thresholds[prob]
                solved = bool(reached.any())
                k = int(np.argmax(reached)) if solved else None  # first position reaching the threshold

                row = {
                    "algo": algo,
                    "problem": prob,
                    "instance": inst,
                    "n_dim": sum(bool(_X_COL.match(c)) for c in i_data.columns),
                    "solved": solved,
                    "fmin": obj.min(),  # best feasible objective at highest fidelity, inf if none
                }

                for name, v in _eval_metrics(metrics, i_data, f"{algo}/{prob}/{inst}").items():
                    row[name] = v[k] if solved else np.inf
                    row[f"max_{name}"] = v[-1]

                rows.append(row)

    return pd.DataFrame(rows)


def _check_metric(table: pd.DataFrame, metric_name: str) -> None:
    if metric_name not in table.columns or f"max_{metric_name}" not in table.columns:
        available = sorted(c[4:] for c in table.columns if c.startswith("max_"))
        raise ValueError(f"Metric '{metric_name}' not in solve table. Available: {available}")


def _ecdf(costs: np.ndarray, n_instance: int, x_start: float, x_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Step ECDF over all instances; non-finite (unsolved) costs never contribute."""
    costs = np.sort(costs[np.isfinite(costs)])
    k = costs.size
    x = np.concatenate(([x_start], costs, [x_end]))
    portion = np.concatenate(([0], np.arange(1, k + 1), [k])) / n_instance
    return x, portion


def get_data_profile_from_table(table: pd.DataFrame, metric_name: str = "budget", normalize_by_dim: bool = False) -> dict:
    r"""
    Data profile of each algorithm from a solve table.

    The data profile of an algorithm is the fraction of (problem, instance) pairs it solves with a cost lower
    than or equal to :math:`\alpha`:

    .. math::
        d(\alpha) = \frac{1}{N} \left| \{ i : t_i \leq \alpha \} \right|,

    where :math:`t_i` is the cost at which pair :math:`i` is solved (:math:`\infty` if never) and :math:`N` is
    the number of pairs.

    Parameters
    ----------
    table : pandas.DataFrame
        Solve table returned by :func:`get_solve_table`.
    metric_name : str, default="budget"
        Metric used as cost. Must have been tabulated in `table`.
    normalize_by_dim : bool, default=False
        If True, the cost is expressed in units of ``n_dim + 1`` (Moré and Wild, 2009).

    Returns
    -------
    dict
        ``profile[algo]`` is a dictionary with the keys:

        - `metric_name` : ndarray, cost values, from 0 to the largest final cost over all algorithms.
        - ``"portion"`` : ndarray, fraction of instances solved at each cost (use a post-step plot).
        - ``"n_instance"`` : int, number of instances of the algorithm.
        - ``"n_solved"`` : int, number of instances solved.
        - ``"min_<metric_name>"`` : float, smallest cost at which an instance is solved (``inf`` if none).

    Raises
    ------
    ValueError
        If `metric_name` is not in `table`, or if the algorithms do not have the same number of instances.

    See Also
    --------
    get_data_profile, get_perf_profile_from_table
    """
    _check_metric(table, metric_name)

    scale = table["n_dim"] + 1 if normalize_by_dim else 1.0  # Moré-Wild: cost in units of n + 1
    cost = table[metric_name] / scale
    x_end = (table[f"max_{metric_name}"] / scale).max()

    counts = table.groupby("algo").size()
    if counts.nunique() > 1:
        raise ValueError(f"Different number of instances between algorithms:\n{counts}")

    out = {}
    for algo, g in table.groupby("algo", sort=False):
        x, portion = _ecdf(cost.loc[g.index].to_numpy(), len(g), 0.0, x_end)
        n_solved = int(g["solved"].sum())
        out[algo] = {
            metric_name: x,
            "portion": portion,
            "n_instance": len(g),
            "n_solved": n_solved,
            f"min_{metric_name}": x[1] if n_solved else np.inf,
        }
    return out


def get_perf_profile_from_table(table: pd.DataFrame, metric_name: str = "budget", unsolved_by_all: str = "keep") -> dict:
    r"""
    Performance profile of each algorithm from a solve table.

    For each (problem, instance) pair, the cost of an algorithm is divided by the lowest cost among all
    algorithms. The performance profile is the fraction of pairs for which this ratio is lower than or equal to
    :math:`r`:

    .. math::
        \rho(r) = \frac{1}{N} \left| \left\{ i : \frac{t_i}{\min_s t_{i,s}} \leq r \right\} \right|.

    Its value at :math:`r = 1` is the fraction of pairs for which the algorithm is the cheapest, and its limit
    is the fraction of pairs it solves (Dolan and Moré, 2002).

    Parameters
    ----------
    table : pandas.DataFrame
        Solve table returned by :func:`get_solve_table`.
    metric_name : str, default="budget"
        Metric used as cost. Must have been tabulated in `table`.
    unsolved_by_all : {"keep", "drop"}, default="keep"
        Policy for the pairs that no algorithm solves. ``"keep"`` counts them as failures for every algorithm,
        so that no profile reaches 1. ``"drop"`` removes them (Dolan and Moré convention).

    Returns
    -------
    dict
        ``profile[algo]`` is a dictionary with the keys:

        - ``"ratio"`` : ndarray, cost ratios, from 1 to the largest finite ratio over all algorithms.
        - ``"portion"`` : ndarray, fraction of pairs within each ratio (use a post-step plot).
        - ``"n_instance"`` : int, number of pairs considered (after applying `unsolved_by_all`).
        - ``"n_solved"`` : int, number of pairs solved by the algorithm.

    Raises
    ------
    ValueError
        If `metric_name` is not in `table`, if `unsolved_by_all` is invalid, or if the algorithms were not run
        on the same (problem, instance) pairs.

    See Also
    --------
    get_perf_profile, get_data_profile_from_table

    Notes
    -----
    If the lowest cost of a pair is 0 (e.g. solved in the initial DOE with the ``"iter"`` metric), the ratio
    is 1 for algorithms whose cost is also 0 and ``inf`` for the others.
    """
    _check_metric(table, metric_name)
    if unsolved_by_all not in ("drop", "keep"):
        raise ValueError("unsolved_by_all must be 'drop' or 'keep'.")

    # pairs x algos matrix; pivot raises on duplicate keys, NaN means an algo lacks a pair
    costs = table.pivot(index=["problem", "instance"], columns="algo", values=metric_name)
    if costs.isna().any().any():
        missing = costs.index[costs.isna().any(axis=1)].tolist()
        raise ValueError(f"Algorithms were not run on the same problem/instance pairs, e.g. {missing[:5]}")

    best = costs.min(axis=1)
    if unsolved_by_all == "drop":
        keep = np.isfinite(best.to_numpy())
        costs, best = costs[keep], best[keep]

    c = costs.to_numpy(dtype=float)
    b = np.broadcast_to(best.to_numpy(dtype=float)[:, None], c.shape)

    ratio = np.full(c.shape, np.inf)
    solved = np.isfinite(c)
    pos = solved & (b > 0)
    ratio[pos] = c[pos] / b[pos]
    ratio[solved & (b == 0) & (c == 0)] = 1.0  # solved for free (e.g. in the initial DOE with iter/idx): 0/0 -> 1

    finite = ratio[np.isfinite(ratio)]
    r_max = finite.max() if finite.size else 1.0
    n_pairs = c.shape[0]

    out = {}
    for j, algo in enumerate(costs.columns):
        x, portion = _ecdf(ratio[:, j], n_pairs, 1.0, r_max)
        out[algo] = {
            "ratio": x,
            "portion": portion,
            "n_instance": n_pairs,
            "n_solved": int(solved[:, j].sum()),
        }
    return out


def _select_metric(metric_name: str, custom_metrics: Mapping[str, Callable] | None):
    if custom_metrics and metric_name in custom_metrics:
        return {metric_name: custom_metrics[metric_name]}
    return [metric_name]  # built-in or a column of the instance DataFrames


def get_data_profile(
    data: dict,
    metric_name: str = "budget",
    tau: float = 0.1,
    ctol: float = 1e-4,
    f0_method: str | None = None,
    custom_metrics: Mapping[str, Callable] | None = None,
    normalize_by_dim: bool = False,
) -> dict:
    """
    Data profile of each algorithm.

    Convenience function chaining :func:`get_solve_table` and :func:`get_data_profile_from_table`.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance] -> pandas.DataFrame``, as returned by
        :func:`~smt_optim.postprocessing.load_data.load_data`.
    metric_name : str, default="budget"
        Cost metric: ``"idx"``, ``"iter"``, ``"budget"``, a column of the instance DataFrames, or a key of
        `custom_metrics`.
    tau : float, default=0.1
        Tolerance in [0, 1] defining the objective threshold to reach (see :func:`compute_threshold`).
    ctol : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= ctol``.
    f0_method : {"max_feasible", "init_min", "init_max"}, optional
        Reference initial value used to define the threshold (see :func:`compute_f0_fstar`).
    custom_metrics : mapping of str to callable, optional
        Custom metrics ``{name: function}``. ``function`` takes the instance DataFrame and returns a cumulative
        (non-decreasing) cost per row.
    normalize_by_dim : bool, default=False
        If True, the cost is expressed in units of ``n_dim + 1``.

    Returns
    -------
    dict
        See :func:`get_data_profile_from_table`.

    Raises
    ------
    ValueError
        See :func:`get_solve_table` and :func:`get_data_profile_from_table`.

    Examples
    --------
    >>> data, _ = load_data("data/multifidelity")
    >>> profile = get_data_profile(data, "budget", tau=0.1)
    >>> for algo, p in profile.items():
    ...     plt.step(p["budget"], p["portion"], where="post", label=algo)

    A custom cost metric:

    >>> cum_time = {"cum_time": lambda df: df["time"].cumsum().to_numpy()}
    >>> profile = get_data_profile(data, "cum_time", custom_metrics=cum_time)
    """
    table = get_solve_table(data, tau, ctol, f0_method, metrics=_select_metric(metric_name, custom_metrics))
    return get_data_profile_from_table(table, metric_name, normalize_by_dim)


def get_perf_profile(
    data: dict,
    metric_name: str = "budget",
    tau: float = 0.1,
    ctol: float = 1e-4,
    f0_method: str | None = None,
    custom_metrics: Mapping[str, Callable] | None = None,
    unsolved_by_all: str = "keep",
) -> dict:
    """
    Performance profile of each algorithm.

    Convenience function chaining :func:`get_solve_table` and :func:`get_perf_profile_from_table`.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance] -> pandas.DataFrame``, as returned by
        :func:`~smt_optim.postprocessing.load_data.load_data`.
    metric_name : str, default="budget"
        Cost metric: ``"idx"``, ``"iter"``, ``"budget"``, a column of the instance DataFrames, or a key of
        `custom_metrics`.
    tau : float, default=0.1
        Tolerance in [0, 1] defining the objective threshold to reach (see :func:`compute_threshold`).
    ctol : float, default=1e-4
        Feasibility tolerance. A point is feasible if ``rscv <= ctol``.
    f0_method : {"max_feasible", "init_min", "init_max"}, optional
        Reference initial value used to define the threshold (see :func:`compute_f0_fstar`).
    custom_metrics : mapping of str to callable, optional
        Custom metrics ``{name: function}``. ``function`` takes the instance DataFrame and returns a cumulative
        (non-decreasing) cost per row.
    unsolved_by_all : {"keep", "drop"}, default="keep"
        Policy for the pairs that no algorithm solves (see :func:`get_perf_profile_from_table`).

    Returns
    -------
    dict
        See :func:`get_perf_profile_from_table`.

    Raises
    ------
    ValueError
        See :func:`get_solve_table` and :func:`get_perf_profile_from_table`.

    Examples
    --------
    >>> data, _ = load_data("data/multifidelity")
    >>> profile = get_perf_profile(data, "budget", tau=0.1)
    >>> for algo, p in profile.items():
    ...     plt.step(p["ratio"], p["portion"], where="post", label=algo)
    >>> plt.xscale("log", base=2)
    """
    table = get_solve_table(data, tau, ctol, f0_method, metrics=_select_metric(metric_name, custom_metrics))
    return get_perf_profile_from_table(table, metric_name, unsolved_by_all)
