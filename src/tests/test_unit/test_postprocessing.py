import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import smt_optim.postprocessing as postprocessing
from smt_optim.postprocessing import (
    get_convergence_profile,
    get_data_profile,
    get_data_profile_from_table,
    get_perf_profile,
    get_perf_profile_from_table,
    get_solve_table,
    load_data,
)
from smt_optim.postprocessing.profiles import (
    compute_f0_fstar,
    compute_threshold,
    order_f0,
)

# sample datasets also used by the postprocessing tutorial
DATA_DIR = Path(__file__).resolve().parents[3] / "examples" / "data"
UNCONSTRAINED_DIR = DATA_DIR / "unconstrained"
MULTIFIDELITY_DIR = DATA_DIR / "multifidelity"


def _df(f0, iters=None, rscv=None, fidelity=None, budget=None, time=None, n_dim=2):
    """Build the DOE of one run. Defaults: all iterations 0..n-1, feasible, fidelity 0, budget 1..n."""
    n = len(f0)
    cols = {
        "iter": np.arange(n) if iters is None else iters,
        "budget": np.arange(1, n + 1) if budget is None else budget,
        "fidelity": np.zeros(n, dtype=int) if fidelity is None else fidelity,
    }
    for d in range(n_dim):
        cols[f"x{d}"] = np.zeros(n)
    cols["f0"] = np.asarray(f0, dtype=float)
    cols["rscv"] = np.zeros(n) if rscv is None else rscv
    if time is not None:
        cols["time"] = time
    return pd.DataFrame(cols)


def _table(costs, max_cost, metric="budget", n_dim=2):
    """
    Build a solve table by hand.

    `costs` maps ``algo -> list of costs`` (one per (problem "P", instance) pair, ``inf`` if unsolved).
    """
    rows = []
    for algo, algo_costs in costs.items():
        for k, c in enumerate(algo_costs):
            rows.append(
                {
                    "algo": algo,
                    "problem": "P",
                    "instance": f"instance_{k}",
                    "n_dim": n_dim,
                    "solved": bool(np.isfinite(c)),
                    "fmin": 0.0,
                    metric: c,
                    f"max_{metric}": max_cost[algo]
                    if isinstance(max_cost, dict)
                    else max_cost,
                }
            )
    return pd.DataFrame(rows)


def _solver_data():
    """
    One problem, one instance. All runs start at 10 and the best value found overall is 0 (fstar=0, f0=10).

    - ``solver`` : 10, 8, 5, 2, 0  -> reaches 5 at row 2 and 1 at row 4
    - ``stuck``  : 10, 9, 9, 9, 9  -> never reaches 5
    """
    n = 5
    return {
        "solver": {
            "P": {
                "instance_0": _df(
                    [10, 8, 5, 2, 0], iters=[0, 0, 1, 2, 3], time=np.ones(n)
                )
            }
        },
        "stuck": {
            "P": {
                "instance_0": _df(
                    [10, 9, 9, 9, 9], iters=[0, 0, 1, 2, 3], time=np.ones(n)
                )
            }
        },
    }


class TestPackageExports(unittest.TestCase):
    def test_all_names_are_importable(self):
        for name in postprocessing.__all__:
            self.assertTrue(hasattr(postprocessing, name), name)

    def test_all_has_no_concatenated_names(self):
        self.assertEqual(len(postprocessing.__all__), len(set(postprocessing.__all__)))
        self.assertIn("get_data_profile", postprocessing.__all__)
        self.assertIn("get_perf_profile", postprocessing.__all__)
        self.assertIn("get_perf_profile_from_table", postprocessing.__all__)

    def test_star_import(self):
        namespace = {}
        exec("from smt_optim.postprocessing import *", namespace)
        for name in postprocessing.__all__:
            self.assertIn(name, namespace)


class TestOrderF0(unittest.TestCase):
    def test_cumulative_minimum_starts_at_best_initial_point(self):
        df = _df([5, 3, 4, 2, 6, 1], iters=[0, 0, 0, 1, 2, 3])
        out = order_f0(df)

        self.assertEqual(list(out.columns), ["iter", "budget", "fmin"])
        np.testing.assert_array_equal(out["iter"], [0, 1, 2, 3])
        np.testing.assert_array_equal(out["budget"], [3, 4, 5, 6])
        np.testing.assert_array_equal(out["fmin"], [3, 2, 2, 1])

    def test_infeasible_points_are_ignored(self):
        df = _df([5, 3, 4, 0, 6, 1], iters=[0, 0, 0, 1, 2, 3], rscv=[0, 0, 0, 1, 0, 0])
        out = order_f0(df)

        np.testing.assert_array_equal(out["fmin"], [3, 3, 3, 1])

    def test_ctol_defines_feasibility(self):
        df = _df([5, 1], iters=[0, 1], rscv=[0, 1e-3])

        np.testing.assert_array_equal(order_f0(df, ctol=1e-4)["fmin"], [5, 5])
        np.testing.assert_array_equal(order_f0(df, ctol=1e-2)["fmin"], [5, 1])

    def test_inf_until_a_feasible_point_is_found(self):
        df = _df([5, 3, 4, 2], iters=[0, 0, 1, 2], rscv=[1, 1, 0, 0])
        out = order_f0(df)

        np.testing.assert_array_equal(out["fmin"], [np.inf, 4, 2])

    def test_only_highest_fidelity_is_considered(self):
        df = _df([1, 1, 5, 3, 4], iters=[0, 0, 0, 1, 2], fidelity=[0, 0, 1, 1, 1])
        out = order_f0(df)

        np.testing.assert_array_equal(out["iter"], [0, 1, 2])
        np.testing.assert_array_equal(out["fmin"], [5, 3, 3])

    def test_custom_metric_column(self):
        df = _df([4, 2, 3], iters=[0, 1, 2], time=[0.5, 1.0, 1.5])
        out = order_f0(df, metric_name="time")

        self.assertEqual(list(out.columns), ["iter", "time", "fmin"])
        np.testing.assert_array_equal(out["time"], [0.5, 1.0, 1.5])

    def test_input_is_not_modified(self):
        df = _df([5, 3, 4, 2, 6, 1], iters=[0, 0, 0, 1, 2, 3])
        before = df.copy()
        order_f0(df)

        pd.testing.assert_frame_equal(df, before)


class TestConvergenceProfile(unittest.TestCase):
    def setUp(self):
        # instance A: budget 1..4, instance B: budget 1, 3, 5
        inst_a = _df([4, 2, 3, 1], iters=[0, 0, 1, 2])
        inst_b = _df([6, 5, 0], iters=[0, 1, 2], budget=[1, 3, 5])
        self.data = {"algo": {"P": {"instance_0": inst_a, "instance_1": inst_b}}}
        with np.errstate(invalid="ignore"):  # std of inf values
            self.conv = get_convergence_profile(self.data)["algo"]["P"]

    def test_structure(self):
        self.assertEqual(
            set(self.conv.keys()),
            {"budget", "max_metric", "f0", "mean", "std", "min", "max"},
        )

    def test_cost_grid_is_the_sorted_union_of_all_instances(self):
        np.testing.assert_array_equal(self.conv["budget"], [1, 2, 3, 4, 5])

    def test_max_metric_is_final_cost_of_each_instance(self):
        np.testing.assert_array_equal(self.conv["max_metric"], [4, 5])

    def test_best_objective_is_carried_forward_on_the_common_grid(self):
        expected = np.array(
            [
                [np.inf, 6],
                [2, 6],
                [2, 5],
                [1, 5],
                [1, 0],
            ]
        )
        np.testing.assert_array_equal(self.conv["f0"], expected)

    def test_best_objective_is_non_increasing(self):
        f0 = self.conv["f0"]
        self.assertTrue(np.all(f0[1:] <= f0[:-1]))

    def test_statistics(self):
        f0 = self.conv["f0"]
        # the first cost has no feasible point for instance A: inf statistics, NaN std
        self.assertEqual(self.conv["mean"][0], np.inf)
        self.assertTrue(np.isnan(self.conv["std"][0]))

        np.testing.assert_allclose(self.conv["mean"][1:], f0[1:].mean(axis=1))
        np.testing.assert_allclose(self.conv["std"][1:], f0[1:].std(axis=1))
        np.testing.assert_allclose(self.conv["mean"][1:], [4, 3.5, 3, 0.5])
        np.testing.assert_allclose(self.conv["std"][1:], [2, 1.5, 2, 0.5])
        np.testing.assert_array_equal(self.conv["min"], f0.min(axis=1))
        np.testing.assert_array_equal(self.conv["max"], f0.max(axis=1))
        self.assertTrue(np.all(self.conv["min"] <= self.conv["max"]))

    def test_other_cost_metric(self):
        with np.errstate(invalid="ignore"):
            conv = get_convergence_profile(self.data, metric_name="iter")["algo"]["P"]

        self.assertIn("iter", conv)
        np.testing.assert_array_equal(conv["iter"], [0, 1, 2])
        np.testing.assert_array_equal(conv["max_metric"], [2, 2])

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            get_convergence_profile(self.data, metric_name="unknown")

    def test_every_algorithm_and_problem_is_profiled(self):
        data = {
            "a": {"P": {"i0": _df([3, 2, 1])}, "Q": {"i0": _df([3, 2, 1])}},
            "b": {"P": {"i0": _df([3, 2, 1])}},
        }
        conv = get_convergence_profile(data)

        self.assertEqual(set(conv.keys()), {"a", "b"})
        self.assertEqual(set(conv["a"].keys()), {"P", "Q"})
        self.assertEqual(set(conv["b"].keys()), {"P"})

    def test_empty_data(self):
        self.assertEqual(get_convergence_profile({}), {})


class TestComputeF0Fstar(unittest.TestCase):
    def setUp(self):
        # algo_a: initial DOE {10, 8}, best 2, worst 12. algo_b: initial DOE {9, 7}, best 1, worst 9
        self.data = {
            "algo_a": {"P": {"instance_0": _df([10, 8, 12, 2], iters=[0, 0, 1, 2])}},
            "algo_b": {"P": {"instance_0": _df([9, 7, 1], iters=[0, 0, 1])}},
        }

    def test_default_is_max_feasible(self):
        res = compute_f0_fstar(self.data)

        self.assertEqual(res["P"], {"fstar": 1, "f0": 12})
        self.assertEqual(compute_f0_fstar(self.data, method="max_feasible"), res)

    def test_init_min(self):
        res = compute_f0_fstar(self.data, method="init_min")

        # largest over the instances of the smallest initial value: max(8, 7)
        self.assertEqual(res["P"], {"fstar": 1, "f0": 8})

    def test_init_max(self):
        res = compute_f0_fstar(self.data, method="init_max")

        # largest over the instances of the largest initial value: max(10, 9)
        self.assertEqual(res["P"], {"fstar": 1, "f0": 10})

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            compute_f0_fstar(self.data, method="unknown")

    def test_infeasible_points_are_ignored(self):
        data = {"a": {"P": {"i0": _df([5, -100, 3], iters=[0, 1, 2], rscv=[0, 1, 0])}}}
        res = compute_f0_fstar(data)

        self.assertEqual(res["P"], {"fstar": 3, "f0": 5})

    def test_eps_defines_feasibility(self):
        data = {"a": {"P": {"i0": _df([5, 1], iters=[0, 1], rscv=[0, 1e-3])}}}

        self.assertEqual(compute_f0_fstar(data, eps=1e-4)["P"]["fstar"], 5)
        self.assertEqual(compute_f0_fstar(data, eps=1e-2)["P"]["fstar"], 1)

    def test_only_highest_fidelity_is_used(self):
        data = {"a": {"P": {"i0": _df([0, 7, 4], iters=[0, 0, 1], fidelity=[0, 1, 1])}}}
        res = compute_f0_fstar(data)

        self.assertEqual(res["P"], {"fstar": 4, "f0": 7})

    def test_problems_are_independent(self):
        data = {"a": {"P": {"i0": _df([5, 1])}, "Q": {"i0": _df([50, 10])}}}
        res = compute_f0_fstar(data)

        self.assertEqual(res["P"], {"fstar": 1, "f0": 5})
        self.assertEqual(res["Q"], {"fstar": 10, "f0": 50})

    def test_problem_without_feasible_point(self):
        data = {"a": {"P": {"i0": _df([5, 1], rscv=[1, 1])}}}
        res = compute_f0_fstar(data)

        self.assertEqual(res["P"], {"fstar": np.inf, "f0": -np.inf})

    def test_init_methods_require_feasible_initial_point(self):
        data = {"a": {"P": {"i0": _df([5, 1], iters=[0, 1], rscv=[1, 0])}}}

        for method in ("init_min", "init_max"):
            with self.assertRaises(ValueError):
                compute_f0_fstar(data, method=method)

        # the default method does not need a feasible initial point
        self.assertEqual(compute_f0_fstar(data)["P"], {"fstar": 1, "f0": 1})


class TestComputeThreshold(unittest.TestCase):
    def test_threshold(self):
        f0_fstar = {"P": {"f0": 10.0, "fstar": 0.0}, "Q": {"f0": 3.0, "fstar": 1.0}}

        self.assertAlmostEqual(compute_threshold(f0_fstar)["P"], 1.0)
        self.assertAlmostEqual(compute_threshold(f0_fstar, tau=0.5)["Q"], 2.0)

    def test_tau_bounds(self):
        f0_fstar = {"P": {"f0": 10.0, "fstar": 2.0}}

        self.assertEqual(compute_threshold(f0_fstar, tau=0.0)["P"], 2.0)
        self.assertEqual(compute_threshold(f0_fstar, tau=1.0)["P"], 10.0)


class TestGetSolveTable(unittest.TestCase):
    def test_columns_and_rows(self):
        table = get_solve_table(_solver_data())

        self.assertEqual(len(table), 2)
        self.assertEqual(
            list(table.columns),
            [
                "algo",
                "problem",
                "instance",
                "n_dim",
                "solved",
                "fmin",
                "idx",
                "max_idx",
                "iter",
                "max_iter",
                "budget",
                "max_budget",
            ],
        )
        self.assertEqual(set(table["algo"]), {"solver", "stuck"})

    def test_solved_run(self):
        table = get_solve_table(_solver_data(), tau=0.1)
        row = table[table["algo"] == "solver"].iloc[0]

        # threshold = 0 + 0.1 * (10 - 0) = 1, first reached at row 4 (f0 = 0)
        self.assertTrue(row["solved"])
        self.assertEqual(row["fmin"], 0)
        self.assertEqual((row["idx"], row["iter"], row["budget"]), (4, 3, 5))
        self.assertEqual(
            (row["max_idx"], row["max_iter"], row["max_budget"]), (4, 3, 5)
        )
        self.assertEqual(row["n_dim"], 2)

    def test_unsolved_run(self):
        table = get_solve_table(_solver_data(), tau=0.1)
        row = table[table["algo"] == "stuck"].iloc[0]

        self.assertFalse(row["solved"])
        self.assertEqual(row["fmin"], 9)
        for metric in ("idx", "iter", "budget"):
            self.assertEqual(row[metric], np.inf)
        self.assertEqual(row["max_budget"], 5)

    def test_first_row_reaching_the_threshold_is_used(self):
        # tau = 0.5 -> threshold 5, reached at row 2 (f0 = 5, equality counts)
        table = get_solve_table(_solver_data(), tau=0.5)
        row = table[table["algo"] == "solver"].iloc[0]

        self.assertEqual((row["idx"], row["iter"], row["budget"]), (2, 1, 3))

    def test_tau_one_is_solved_by_the_initial_point(self):
        table = get_solve_table(_solver_data(), tau=1.0)

        self.assertTrue(table["solved"].all())
        self.assertTrue((table["idx"] == 0).all())

    def test_infeasible_points_do_not_solve(self):
        data = _solver_data()
        data["infeasible"] = {
            "P": {
                "instance_0": _df(
                    [10, 0, 0, 0, 0], iters=[0, 0, 1, 2, 3], rscv=[0, 1, 1, 1, 1]
                )
            }
        }
        table = get_solve_table(data, tau=0.1)
        row = table[table["algo"] == "infeasible"].iloc[0]

        self.assertFalse(row["solved"])
        self.assertEqual(row["fmin"], 10)

    def test_low_fidelity_points_do_not_solve(self):
        data = _solver_data()
        data["lofi"] = {
            "P": {
                "instance_0": _df(
                    [10, 0, 0, 0, 0], iters=[0, 0, 1, 2, 3], fidelity=[1, 0, 0, 0, 0]
                )
            }
        }
        table = get_solve_table(data, tau=0.1)
        row = table[table["algo"] == "lofi"].iloc[0]

        self.assertFalse(row["solved"])
        self.assertEqual(row["fmin"], 10)

    def test_ctol(self):
        # "ref" fixes fstar = 0; "a" only reaches 0 with rscv = 1e-3
        data = {
            "ref": {"P": {"i0": _df([10, 0], iters=[0, 1])}},
            "a": {"P": {"i0": _df([10, 0], iters=[0, 1], rscv=[0, 1e-3])}},
        }

        def solved(ctol):
            table = get_solve_table(data, tau=0.0, ctol=ctol)
            return table[table["algo"] == "a"]["solved"].iloc[0]

        self.assertFalse(solved(1e-4))
        self.assertTrue(solved(1e-2))

    def test_f0_method_changes_the_threshold(self):
        # fstar = 2 and tau = 0.5. Reference f0: max_feasible 12, init_max 10, init_min 8
        # -> thresholds 7, 6 and 5. A value of 6.5 solves only with max_feasible.
        data = {"a": {"P": {"i0": _df([10, 8, 12, 6.5, 2], iters=[0, 0, 1, 2, 3])}}}
        expected = {None: 3, "max_feasible": 3, "init_max": 4, "init_min": 4}

        for method, idx in expected.items():
            table = get_solve_table(data, tau=0.5, f0_method=method)
            self.assertEqual(table["idx"].iloc[0], idx, msg=method)

    def test_n_dim_is_the_number_of_x_columns(self):
        data = {"a": {"P": {"i0": _df([2, 1], n_dim=5)}}}

        self.assertEqual(get_solve_table(data)["n_dim"].iloc[0], 5)

    def test_one_row_per_algo_problem_instance(self):
        data = {
            algo: {
                prob: {inst: _df([3, 2, 1]) for inst in ("i0", "i1")}
                for prob in ("P", "Q")
            }
            for algo in ("a", "b")
        }
        table = get_solve_table(data)

        self.assertEqual(len(table), 8)
        self.assertEqual(len(table.drop_duplicates(["algo", "problem", "instance"])), 8)

    def test_thresholds_are_shared_between_algorithms(self):
        # the best value of "solver" sets fstar for "late" as well
        data = {
            "solver": {"P": {"i0": _df([10, 0])}},
            "late": {"P": {"i0": _df([10, 5, 3])}},
        }
        table = get_solve_table(data, tau=0.1)

        self.assertTrue(table[table["algo"] == "solver"]["solved"].iloc[0])
        self.assertFalse(table[table["algo"] == "late"]["solved"].iloc[0])


class TestSolveTableMetrics(unittest.TestCase):
    def test_sequence_of_metrics(self):
        table = get_solve_table(_solver_data(), metrics=["budget"])

        self.assertIn("budget", table.columns)
        self.assertIn("max_budget", table.columns)
        self.assertNotIn("iter", table.columns)

    def test_dataframe_column_as_metric(self):
        table = get_solve_table(_solver_data(), metrics=["time"])
        row = table[table["algo"] == "solver"].iloc[0]

        self.assertEqual(
            row["time"], 1.0
        )  # DataFrame column used as is: value at the solving row
        self.assertEqual(row["max_time"], 1.0)

    def test_mapping_of_custom_metrics(self):
        metrics = {"cum_time": lambda df: df["time"].cumsum().to_numpy()}
        table = get_solve_table(_solver_data(), metrics=metrics)
        solver = table[table["algo"] == "solver"].iloc[0]
        stuck = table[table["algo"] == "stuck"].iloc[0]

        self.assertEqual(solver["cum_time"], 5)
        self.assertEqual(solver["max_cum_time"], 5)
        self.assertEqual(stuck["cum_time"], np.inf)
        self.assertEqual(stuck["max_cum_time"], 5)

    def test_reserved_names_raise(self):
        for name in (
            "algo",
            "problem",
            "instance",
            "n_dim",
            "solved",
            "fmin",
            "max_foo",
        ):
            with self.assertRaises(ValueError, msg=name):
                get_solve_table(
                    _solver_data(), metrics={name: lambda df: np.arange(len(df))}
                )

    def test_reserved_name_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            get_solve_table(_solver_data(), metrics=["solved"])

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            get_solve_table(_solver_data(), metrics=["unknown"])

    def test_decreasing_metric_raises(self):
        with self.assertRaises(ValueError):
            get_solve_table(
                _solver_data(), metrics={"m": lambda df: -np.arange(len(df))}
            )

    def test_nan_metric_raises(self):
        with self.assertRaises(ValueError):
            get_solve_table(
                _solver_data(), metrics={"m": lambda df: np.full(len(df), np.nan)}
            )

    def test_wrong_length_metric_raises(self):
        with self.assertRaises(ValueError):
            get_solve_table(_solver_data(), metrics={"m": lambda df: np.zeros(3)})


class TestDataProfileFromTable(unittest.TestCase):
    def setUp(self):
        self.table = _table(
            {
                "a": [2, 4, np.inf, 6],
                "b": [np.inf, 3, 8, np.inf],
                "none": [np.inf] * 4,
            },
            max_cost={"a": 10, "b": 12, "none": 5},
        )

    def test_ecdf_values(self):
        profile = get_data_profile_from_table(self.table, "budget")

        # costs are stepped on [0, largest final cost over all algorithms] = [0, 12]
        np.testing.assert_array_equal(profile["a"]["budget"], [0, 2, 4, 6, 12])
        np.testing.assert_allclose(profile["a"]["portion"], [0, 0.25, 0.5, 0.75, 0.75])
        np.testing.assert_array_equal(profile["b"]["budget"], [0, 3, 8, 12])
        np.testing.assert_allclose(profile["b"]["portion"], [0, 0.25, 0.5, 0.5])

    def test_counts_and_min_cost(self):
        profile = get_data_profile_from_table(self.table, "budget")

        self.assertEqual(profile["a"]["n_instance"], 4)
        self.assertEqual(profile["a"]["n_solved"], 3)
        self.assertEqual(profile["a"]["min_budget"], 2)
        self.assertEqual(profile["b"]["n_solved"], 2)
        self.assertEqual(profile["b"]["min_budget"], 3)

    def test_algorithm_that_solves_nothing(self):
        profile = get_data_profile_from_table(self.table, "budget")

        self.assertEqual(profile["none"]["n_solved"], 0)
        self.assertEqual(profile["none"]["min_budget"], np.inf)
        np.testing.assert_array_equal(profile["none"]["portion"], [0, 0])

    def test_portion_is_a_monotone_cdf(self):
        for p in get_data_profile_from_table(self.table, "budget").values():
            self.assertEqual(p["portion"][0], 0)
            self.assertTrue(np.all(np.diff(p["portion"]) >= 0))
            self.assertTrue(np.all(np.diff(p["budget"]) >= 0))
            self.assertAlmostEqual(p["portion"][-1], p["n_solved"] / p["n_instance"])
            self.assertEqual(p["budget"][0], 0)
            self.assertEqual(p["budget"][-1], 12)

    def test_normalize_by_dim(self):
        profile = get_data_profile_from_table(
            self.table, "budget", normalize_by_dim=True
        )

        # n_dim = 2 -> costs in units of n_dim + 1 = 3
        np.testing.assert_allclose(profile["a"]["budget"], [0, 2 / 3, 4 / 3, 2, 4])
        np.testing.assert_allclose(profile["a"]["portion"], [0, 0.25, 0.5, 0.75, 0.75])
        self.assertAlmostEqual(profile["a"]["min_budget"], 2 / 3)

    def test_normalize_by_dim_uses_each_instance_dimension(self):
        table = _table({"a": [2.0, 4.0]}, max_cost=10)
        table["n_dim"] = [1, 3]
        profile = get_data_profile_from_table(table, "budget", normalize_by_dim=True)

        np.testing.assert_allclose(profile["a"]["budget"][1:-1], [1.0, 1.0])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            get_data_profile_from_table(self.table, "unknown")

    def test_unequal_number_of_instances_raises(self):
        table = self.table.drop(index=0)

        with self.assertRaises(ValueError):
            get_data_profile_from_table(table, "budget")

    def test_other_metric(self):
        table = _table({"a": [1, 2], "b": [2, np.inf]}, max_cost=4, metric="iter")
        profile = get_data_profile_from_table(table, "iter")

        self.assertIn("iter", profile["a"])
        self.assertIn("min_iter", profile["a"])
        np.testing.assert_array_equal(profile["a"]["iter"], [0, 1, 2, 4])


class TestPerfProfileFromTable(unittest.TestCase):
    def setUp(self):
        # best cost per pair: [2, 4, 8, 6]
        self.table = _table(
            {"a": [2, 4, np.inf, 6], "b": [2, 6, 8, np.inf]}, max_cost=10
        )

    def test_ratio_and_portion(self):
        profile = get_perf_profile_from_table(self.table, "budget")

        # ratios: a = [1, 1, inf, 1]; b = [1, 1.5, 1, inf]
        np.testing.assert_allclose(profile["a"]["ratio"], [1, 1, 1, 1, 1.5])
        np.testing.assert_allclose(profile["a"]["portion"], [0, 0.25, 0.5, 0.75, 0.75])
        np.testing.assert_allclose(profile["b"]["ratio"], [1, 1, 1, 1.5, 1.5])
        np.testing.assert_allclose(profile["b"]["portion"], [0, 0.25, 0.5, 0.75, 0.75])

    def test_counts(self):
        profile = get_perf_profile_from_table(self.table, "budget")

        for p in profile.values():
            self.assertEqual(p["n_instance"], 4)
            self.assertEqual(p["n_solved"], 3)

    def test_portion_is_bounded_by_the_solved_fraction(self):
        for p in get_perf_profile_from_table(self.table, "budget").values():
            self.assertEqual(p["ratio"][0], 1)
            self.assertTrue(np.all(np.diff(p["ratio"]) >= 0))
            self.assertTrue(np.all(np.diff(p["portion"]) >= 0))
            self.assertAlmostEqual(p["portion"][-1], p["n_solved"] / p["n_instance"])

    def test_cheapest_algorithm_has_ratio_one(self):
        table = _table({"fast": [1, 1], "slow": [4, 3]}, max_cost=10)
        profile = get_perf_profile_from_table(table, "budget")

        np.testing.assert_allclose(profile["fast"]["ratio"], [1, 1, 1, 4])
        np.testing.assert_allclose(profile["fast"]["portion"], [0, 0.5, 1, 1])
        np.testing.assert_allclose(profile["slow"]["ratio"], [1, 3, 4, 4])
        np.testing.assert_allclose(profile["slow"]["portion"], [0, 0.5, 1, 1])

    def test_pairs_unsolved_by_all_are_kept_by_default(self):
        table = _table({"a": [2, 4, np.inf], "b": [3, 4, np.inf]}, max_cost=10)
        profile = get_perf_profile_from_table(table, "budget")

        for p in profile.values():
            self.assertEqual(p["n_instance"], 3)
            self.assertEqual(p["n_solved"], 2)
            self.assertAlmostEqual(p["portion"][-1], 2 / 3)

    def test_pairs_unsolved_by_all_can_be_dropped(self):
        table = _table({"a": [2, 4, np.inf], "b": [3, 4, np.inf]}, max_cost=10)
        profile = get_perf_profile_from_table(table, "budget", unsolved_by_all="drop")

        for p in profile.values():
            self.assertEqual(p["n_instance"], 2)
            self.assertEqual(p["n_solved"], 2)
            self.assertAlmostEqual(p["portion"][-1], 1.0)

    def test_invalid_unsolved_by_all_raises(self):
        with self.assertRaises(ValueError):
            get_perf_profile_from_table(self.table, "budget", unsolved_by_all="ignore")

    def test_nothing_solved(self):
        table = _table({"a": [np.inf, np.inf], "b": [np.inf, np.inf]}, max_cost=10)
        profile = get_perf_profile_from_table(table, "budget")

        for p in profile.values():
            self.assertEqual(p["n_solved"], 0)
            np.testing.assert_array_equal(p["ratio"], [1, 1])
            np.testing.assert_array_equal(p["portion"], [0, 0])

    def test_zero_cost(self):
        # pair 0: solved for free by both; pair 1: solved for free by "b" only
        table = _table({"a": [0, 3], "b": [0, 0]}, max_cost=5, metric="iter")
        profile = get_perf_profile_from_table(table, "iter")

        self.assertEqual(profile["a"]["n_solved"], 2)
        self.assertAlmostEqual(
            profile["a"]["portion"][-1], 0.5
        )  # ratio 1 on pair 0, inf on pair 1
        self.assertAlmostEqual(profile["b"]["portion"][-1], 1.0)

    def test_mismatched_pairs_raise(self):
        table = self.table.drop(index=self.table.index[-1])

        with self.assertRaises(ValueError):
            get_perf_profile_from_table(table, "budget")

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            get_perf_profile_from_table(self.table, "unknown")


class TestProfileConvenienceFunctions(unittest.TestCase):
    def test_data_profile_chains_table_and_profile(self):
        data = _solver_data()
        expected = get_data_profile_from_table(
            get_solve_table(data, tau=0.1, metrics=["budget"]), "budget"
        )
        profile = get_data_profile(data, "budget", tau=0.1)

        self.assertEqual(set(profile), set(expected))
        for algo in expected:
            np.testing.assert_array_equal(
                profile[algo]["budget"], expected[algo]["budget"]
            )
            np.testing.assert_array_equal(
                profile[algo]["portion"], expected[algo]["portion"]
            )

    def test_data_profile_values(self):
        profile = get_data_profile(_solver_data(), "budget", tau=0.1)

        np.testing.assert_array_equal(profile["solver"]["budget"], [0, 5, 5])
        np.testing.assert_array_equal(profile["solver"]["portion"], [0, 1, 1])
        np.testing.assert_array_equal(profile["stuck"]["portion"], [0, 0])

    def test_perf_profile_chains_table_and_profile(self):
        data = _solver_data()
        expected = get_perf_profile_from_table(
            get_solve_table(data, tau=0.1, metrics=["budget"]), "budget"
        )
        profile = get_perf_profile(data, "budget", tau=0.1)

        for algo in expected:
            np.testing.assert_array_equal(
                profile[algo]["ratio"], expected[algo]["ratio"]
            )
            np.testing.assert_array_equal(
                profile[algo]["portion"], expected[algo]["portion"]
            )

    def test_perf_profile_values(self):
        profile = get_perf_profile(_solver_data(), "budget", tau=0.1)

        self.assertEqual(profile["solver"]["n_solved"], 1)
        self.assertEqual(profile["solver"]["portion"][-1], 1.0)
        self.assertEqual(profile["stuck"]["n_solved"], 0)
        self.assertEqual(profile["stuck"]["portion"][-1], 0.0)

    def test_custom_metric(self):
        custom = {"cum_time": lambda df: df["time"].cumsum().to_numpy()}

        data_profile = get_data_profile(
            _solver_data(), "cum_time", custom_metrics=custom
        )
        perf_profile = get_perf_profile(
            _solver_data(), "cum_time", custom_metrics=custom
        )

        self.assertIn("cum_time", data_profile["solver"])
        np.testing.assert_array_equal(data_profile["solver"]["cum_time"], [0, 5, 5])
        self.assertEqual(perf_profile["solver"]["n_solved"], 1)

    def test_dataframe_column_metric_without_custom_metrics(self):
        profile = get_data_profile(_solver_data(), "time")

        self.assertIn("time", profile["solver"])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            get_data_profile(_solver_data(), "unknown")
        with self.assertRaises(ValueError):
            get_perf_profile(_solver_data(), "unknown")

    def test_normalize_by_dim(self):
        profile = get_data_profile(_solver_data(), "budget", normalize_by_dim=True)

        np.testing.assert_allclose(profile["solver"]["budget"], [0, 5 / 3, 5 / 3])

    def test_unsolved_by_all(self):
        data = {
            "a": {"P": {"i0": _df([10, 0]), "i1": _df([10, 10])}},
            "b": {"P": {"i0": _df([10, 0]), "i1": _df([10, 10])}},
        }
        keep = get_perf_profile(data, "budget", unsolved_by_all="keep")
        drop = get_perf_profile(data, "budget", unsolved_by_all="drop")

        self.assertEqual(keep["a"]["n_instance"], 2)
        self.assertEqual(drop["a"]["n_instance"], 1)


def _check_profiles(test, data):
    """Invariants shared by every profile computed on a real dataset."""
    n_algo = len(data)

    for metric in ("idx", "iter", "budget"):
        data_profile = get_data_profile(data, metric)
        perf_profile = get_perf_profile(data, metric)

        test.assertEqual(set(data_profile), set(data))
        test.assertEqual(set(perf_profile), set(data))
        test.assertEqual(n_algo, len(data_profile))

        for p in data_profile.values():
            test.assertTrue(np.all((p["portion"] >= 0) & (p["portion"] <= 1)))
            test.assertTrue(np.all(np.diff(p["portion"]) >= 0))
            test.assertTrue(np.all(np.diff(p[metric]) >= 0))
            test.assertAlmostEqual(p["portion"][-1], p["n_solved"] / p["n_instance"])

        for p in perf_profile.values():
            test.assertTrue(np.all((p["portion"] >= 0) & (p["portion"] <= 1)))
            test.assertTrue(np.all(np.diff(p["portion"]) >= 0))
            test.assertGreaterEqual(p["ratio"][0], 1)
            test.assertTrue(np.all(np.diff(p["ratio"]) >= 0))


@unittest.skipUnless(UNCONSTRAINED_DIR.is_dir(), "unconstrained example data not found")
class TestUnconstrainedExampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.report = load_data(str(UNCONSTRAINED_DIR))

    def test_data_is_valid(self):
        self.assertTrue(self.report.is_valid)

    def test_solve_table(self):
        table = get_solve_table(self.data)

        # 2 algorithms x 3 problems x 5 instances
        self.assertEqual(len(table), 30)
        self.assertTrue((table["n_dim"] == 2).all())
        self.assertTrue(np.isfinite(table["fmin"]).all())

        solved = table[table["solved"]]
        self.assertTrue(np.isfinite(solved["budget"]).all())
        self.assertTrue((solved["budget"] <= solved["max_budget"]).all())
        self.assertTrue((table[~table["solved"]]["budget"] == np.inf).all())

    def test_profiles(self):
        _check_profiles(self, self.data)

    def test_looser_tolerance_solves_more(self):
        strict = get_solve_table(self.data, tau=0.01)["solved"].sum()
        loose = get_solve_table(self.data, tau=0.5)["solved"].sum()

        self.assertGreaterEqual(loose, strict)

    def test_profiles_from_a_shared_table(self):
        table = get_solve_table(self.data, metrics=["budget", "iter"])

        self.assertEqual(
            set(get_data_profile_from_table(table, "iter")), set(self.data)
        )
        self.assertEqual(
            set(get_perf_profile_from_table(table, "budget")), set(self.data)
        )

    def test_convergence_profile(self):
        for metric in ("budget", "iter"):
            with np.errstate(invalid="ignore"):
                conv = get_convergence_profile(self.data, metric_name=metric)

            self.assertEqual(set(conv), set(self.data))
            for algo, algo_data in self.data.items():
                self.assertEqual(set(conv[algo]), set(algo_data))
                for problem, c in conv[algo].items():
                    n_instance = len(algo_data[problem])
                    f0 = c["f0"]

                    self.assertEqual(f0.shape, (c[metric].size, n_instance))
                    self.assertEqual(c["max_metric"].shape, (n_instance,))
                    self.assertTrue(np.all(np.diff(c[metric]) > 0))
                    self.assertTrue(np.all(f0[1:] <= f0[:-1]))
                    self.assertTrue(np.all(c["min"] <= c["max"]))


@unittest.skipUnless(MULTIFIDELITY_DIR.is_dir(), "multifidelity example data not found")
class TestMultifidelityExampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.report = load_data(str(MULTIFIDELITY_DIR))

    def test_data_is_valid(self):
        self.assertTrue(self.report.is_valid)

    def test_profiles(self):
        _check_profiles(self, self.data)

    def test_constrained_problem_is_handled(self):
        # Gano is constrained: some points are infeasible and must not count
        gano = pd.concat(df for a in self.data.values() for df in a["Gano"].values())
        self.assertTrue((gano["rscv"] > 1e-4).any())

        table = get_solve_table(self.data)
        for algo, algo_data in self.data.items():
            for inst, df in algo_data["Gano"].items():
                feas = df[
                    (df["rscv"] <= 1e-4) & (df["fidelity"] == df["fidelity"].max())
                ]
                row = table[
                    (table["algo"] == algo)
                    & (table["problem"] == "Gano")
                    & (table["instance"] == inst)
                ].iloc[0]
                expected = feas["f0"].min() if len(feas) else np.inf
                self.assertEqual(row["fmin"], expected)

    def test_custom_cumulative_time_metric(self):
        custom = {"cum_time": lambda df: df["time"].cumsum().to_numpy()}
        data_profile = get_data_profile(self.data, "cum_time", custom_metrics=custom)
        perf_profile = get_perf_profile(self.data, "cum_time", custom_metrics=custom)

        for p in data_profile.values():
            self.assertIn("cum_time", p)
            self.assertTrue(np.all(np.diff(p["portion"]) >= 0))
        for p in perf_profile.values():
            self.assertTrue(np.all(np.diff(p["portion"]) >= 0))

    def test_convergence_profile(self):
        with np.errstate(invalid="ignore"):
            conv = get_convergence_profile(self.data)

        for algo, algo_data in self.data.items():
            for problem in algo_data:
                f0 = conv[algo][problem]["f0"]
                self.assertTrue(np.all(f0[1:] <= f0[:-1]))


if __name__ == "__main__":
    unittest.main()
