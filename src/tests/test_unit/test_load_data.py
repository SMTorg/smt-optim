import os
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from smt_optim.postprocessing.load_data import load_data

# shared, valid sample dataset also used by the postprocessing tutorial
VALID_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "data" / "unconstrained"
)


def _write_doe(root, algorithm, problem, instance, rows):
    out_dir = os.path.join(root, algorithm, problem, instance)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "doe.csv"), index=False)


def _default_rows(f0_0=13.0, n_iters=3):
    return [
        dict(
            iter=i,
            budget=i + 1,
            fidelity=0,
            x0=0.0,
            x1=0.0,
            f0=f0_0 / (i + 1),
            rscv=0.0,
        )
        for i in range(n_iters)
    ]


class TestLoadDataValid(unittest.TestCase):
    def test_valid_dataset_loads_without_issues(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data, report = load_data(str(VALID_DATA_DIR))

        self.assertTrue(report.is_valid)
        self.assertEqual(report.summary(), "No issues found.")

        self.assertEqual(
            set(data.keys()), {"expected_improvement", "probability_of_improvement"}
        )
        for algo_data in data.values():
            self.assertEqual(
                set(algo_data.keys()),
                {"Sphere_2d", "SumSquares_2d", "RotatedHyperEllipsoid_2d"},
            )
            for problem_data in algo_data.values():
                self.assertEqual(
                    set(problem_data.keys()), {f"instance_{i}" for i in range(5)}
                )
                for df in problem_data.values():
                    self.assertTrue(
                        {"iter", "budget", "f0", "rscv"}.issubset(df.columns)
                    )

    def test_empty_root_dir_returns_empty_data_and_valid_report(self):
        with tempfile.TemporaryDirectory() as root:
            data, report = load_data(root)

        self.assertEqual(data, {})
        self.assertTrue(report.is_valid)

    def test_algorithms_prefixed_with_underscore_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            rows = _default_rows()
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", rows)
            _write_doe(root, "_archived_cobyla", "Sphere_2d", "instance_0", rows)

            data, report = load_data(root)

        self.assertEqual(set(data.keys()), {"cobyla"})
        self.assertTrue(report.is_valid)

    def test_instance_without_doe_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", _default_rows())
            os.makedirs(os.path.join(root, "cobyla", "Sphere_2d", "instance_1"))

            data, report = load_data(root)

        self.assertEqual(set(data["cobyla"]["Sphere_2d"].keys()), {"instance_0"})


class TestLoadDataMissingColumns(unittest.TestCase):
    def test_missing_required_column_is_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [
                {k: v for k, v in r.items() if k != "rscv"} for r in _default_rows()
            ]
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", rows)

            with self.assertWarns(UserWarning):
                data, report = load_data(root)

        self.assertIn(("cobyla", "Sphere_2d", "instance_0"), report.missing_columns)
        self.assertEqual(
            report.missing_columns[("cobyla", "Sphere_2d", "instance_0")], {"rscv"}
        )
        self.assertNotIn("instance_0", data.get("cobyla", {}).get("Sphere_2d", {}))


class TestLoadDataCoverage(unittest.TestCase):
    def test_missing_problem_instance_combination_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            rows = _default_rows()
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", rows)
            _write_doe(root, "cobyla", "Sphere_2d", "instance_1", rows)
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", rows)

            with self.assertWarns(UserWarning):
                data, report = load_data(root)

        self.assertEqual(
            report.missing_combinations, {"cobyqa": {("Sphere_2d", "instance_1")}}
        )


class TestLoadDataInitialPoint(unittest.TestCase):
    def test_inconsistent_initial_point_is_reported_but_data_still_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(
                root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0)
            )
            _write_doe(
                root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0)
            )

            with self.assertWarns(UserWarning):
                data, report = load_data(root)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)
        self.assertIn("instance_0", data["cobyla"]["Sphere_2d"])
        self.assertIn("instance_0", data["cobyqa"]["Sphere_2d"])

    def test_initial_point_within_tolerance_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(
                root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0)
            )
            _write_doe(
                root,
                "cobyqa",
                "Sphere_2d",
                "instance_0",
                _default_rows(f0_0=13.0 + 1e-10),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                data, report = load_data(root, atol=1e-6)

        self.assertEqual(report.inconsistent_start, {})
        self.assertTrue(report.is_valid)


class TestLoadDataInitialDoe(unittest.TestCase):
    @staticmethod
    def _rows(f0_init, n_iters=2):
        """Initial DOE with one point per value of `f0_init`, followed by `n_iters` iterations."""
        rows = [
            dict(iter=0, budget=k + 1, fidelity=0, x0=0.0, x1=0.0, f0=f0, rscv=0.0)
            for k, f0 in enumerate(f0_init)
        ]
        rows += [
            dict(
                iter=i,
                budget=len(f0_init) + i,
                fidelity=0,
                x0=0.0,
                x1=0.0,
                f0=1.0 / i,
                rscv=0.0,
            )
            for i in range(1, n_iters + 1)
        ]
        return rows

    def _load(self, rows_a, rows_b, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", rows_a)
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", rows_b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return load_data(root, **kwargs)

    def test_difference_in_a_later_initial_point_is_detected(self):
        # same first point, different second point
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), self._rows([5.0, 3.5, 4.0]))

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)
        self.assertFalse(report.is_valid)

    def test_same_initial_doe_in_a_different_order_is_consistent(self):
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), self._rows([4.0, 5.0, 3.0]))

        self.assertEqual(report.inconsistent_start, {})
        self.assertTrue(report.is_valid)

    def test_different_number_of_initial_points_is_detected(self):
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), self._rows([5.0, 3.0]))

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_rscv_of_a_later_initial_point_is_compared(self):
        rows_b = self._rows([5.0, 3.0, 4.0])
        rows_b[2]["rscv"] = 1.0
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_rscv_and_f0_are_compared_per_point(self):
        # same f0 and rscv sets, but paired differently
        rows_a = self._rows([5.0, 3.0])
        rows_b = self._rows([5.0, 3.0])
        rows_a[0]["rscv"], rows_a[1]["rscv"] = 1.0, 0.0
        rows_b[0]["rscv"], rows_b[1]["rscv"] = 0.0, 1.0
        _, report = self._load(rows_a, rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_tolerance_applies_to_all_initial_points(self):
        rows_b = self._rows([5.0, 3.0, 4.0])
        rows_b[2]["f0"] += 1e-10

        _, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b, atol=1e-6)
        self.assertEqual(report.inconsistent_start, {})

        _, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b, atol=1e-12)
        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_design_variables_of_a_later_initial_point_are_compared(self):
        rows_b = self._rows([5.0, 3.0, 4.0])
        rows_b[2]["x1"] = 0.5
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_same_f0_with_different_design_variables_is_detected(self):
        rows_a = self._rows([5.0, 3.0])
        rows_b = self._rows([5.0, 3.0])
        rows_b[0]["x0"] = 1.0
        _, report = self._load(rows_a, rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_same_initial_doe_in_a_different_order_is_consistent_with_x(self):
        rows_a = self._rows([5.0, 3.0, 4.0])
        for k, row in enumerate(rows_a[:3]):
            row["x0"], row["x1"] = float(k), float(10 - k)
        rows_b = [rows_a[2], rows_a[0], rows_a[1]] + rows_a[3:]
        _, report = self._load(rows_a, rows_b)

        self.assertEqual(report.inconsistent_start, {})

    def test_different_number_of_design_variables_is_detected(self):
        rows_b = [{**r, "x2": 0.0} for r in self._rows([5.0, 3.0, 4.0])]
        data, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)
        values = report.inconsistent_start[("Sphere_2d", "instance_0")]
        self.assertEqual(values["cobyla"].shape[1] - 2, 2)
        self.assertEqual(values["cobyqa"].shape[1] - 2, 3)
        self.assertIn("3 variables", report.summary())
        self.assertIn("instance_0", data["cobyqa"]["Sphere_2d"])  # data is kept

    def test_missing_design_variable_columns_are_detected(self):
        rows_b = [
            {k: v for k, v in r.items() if k not in ("x0", "x1")}
            for r in self._rows([5.0, 3.0, 4.0])
        ]
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), rows_b)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)

    def test_design_variables_are_ordered_by_index(self):
        # x10 must come after x2, whatever the column order of the file
        def rows(order):
            return [
                {k: r[k] for k in order}
                for r in (
                    dict(
                        iter=0,
                        budget=1,
                        fidelity=0,
                        x0=0.0,
                        x2=2.0,
                        x10=10.0,
                        f0=1.0,
                        rscv=0.0,
                    ),
                    dict(
                        iter=1,
                        budget=2,
                        fidelity=0,
                        x0=0.0,
                        x2=2.0,
                        x10=10.0,
                        f0=0.5,
                        rscv=0.0,
                    ),
                )
            ]

        order_a = ["iter", "budget", "fidelity", "x0", "x2", "x10", "f0", "rscv"]
        order_b = ["iter", "budget", "fidelity", "x10", "x2", "x0", "f0", "rscv"]
        _, report = self._load(rows(order_a), rows(order_b))

        self.assertEqual(report.inconsistent_start, {})

    def test_report_stores_sorted_initial_doe_and_summary_lists_it(self):
        _, report = self._load(self._rows([5.0, 3.0, 4.0]), self._rows([5.0, 3.5, 4.0]))
        values = report.inconsistent_start[("Sphere_2d", "instance_0")]

        self.assertEqual(values["cobyla"][:, -2].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(values["cobyqa"][:, -2].tolist(), [3.5, 4.0, 5.0])
        self.assertIn("3 initial points", report.summary())


class TestLoadDataStrictMode(unittest.TestCase):
    def test_strict_mode_raises_on_invalid_data(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(
                root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0)
            )
            _write_doe(
                root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0)
            )

            with self.assertRaises(ValueError):
                load_data(root, strict=True)

    def test_non_strict_mode_only_warns(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(
                root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0)
            )
            _write_doe(
                root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0)
            )

            with self.assertWarns(UserWarning):
                data, report = load_data(root, strict=False)

        self.assertFalse(report.is_valid)


if __name__ == "__main__":
    unittest.main()
