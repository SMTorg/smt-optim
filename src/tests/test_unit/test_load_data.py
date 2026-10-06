import os
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from smt_optim.postprocessing.load_data import load_data

# shared, valid sample dataset also used by the postprocessing tutorial
VALID_DATA_DIR = Path(__file__).resolve().parents[3] / "examples" / "data" / "opt_results"


def _write_doe(root, algorithm, problem, instance, rows):
    out_dir = os.path.join(root, algorithm, problem, instance)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "doe.csv"), index=False)


def _default_rows(f0_0=13.0, n_iters=3):
    return [
        dict(iter=i, budget=i + 1, fidelity=0, x0=0.0, x1=0.0, f0=f0_0 / (i + 1), rscv=0.0)
        for i in range(n_iters)
    ]


class TestLoadDataValid(unittest.TestCase):
    def test_valid_dataset_loads_without_issues(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data, report = load_data(str(VALID_DATA_DIR))

        self.assertTrue(report.is_valid)
        self.assertEqual(report.summary(), "No issues found.")

        self.assertEqual(set(data.keys()), {"cobyla", "cobyqa"})
        for algo_data in data.values():
            self.assertEqual(set(algo_data.keys()), {"Sphere_2d", "Rastrigin_2d"})
            for problem_data in algo_data.values():
                self.assertEqual(set(problem_data.keys()), {"instance_0", "instance_1"})
                for df in problem_data.values():
                    self.assertTrue({"iter", "budget", "f0", "rscv"}.issubset(df.columns))

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
            rows = [{k: v for k, v in r.items() if k != "rscv"} for r in _default_rows()]
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", rows)

            with self.assertWarns(UserWarning):
                data, report = load_data(root)

        self.assertIn(("cobyla", "Sphere_2d", "instance_0"), report.missing_columns)
        self.assertEqual(report.missing_columns[("cobyla", "Sphere_2d", "instance_0")], {"rscv"})
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

        self.assertEqual(report.missing_combinations, {"cobyqa": {("Sphere_2d", "instance_1")}})


class TestLoadDataInitialPoint(unittest.TestCase):
    def test_inconsistent_initial_point_is_reported_but_data_still_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0))
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0))

            with self.assertWarns(UserWarning):
                data, report = load_data(root)

        self.assertIn(("Sphere_2d", "instance_0"), report.inconsistent_start)
        self.assertIn("instance_0", data["cobyla"]["Sphere_2d"])
        self.assertIn("instance_0", data["cobyqa"]["Sphere_2d"])

    def test_initial_point_within_tolerance_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0))
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0 + 1e-10))

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                data, report = load_data(root, atol=1e-6)

        self.assertEqual(report.inconsistent_start, {})
        self.assertTrue(report.is_valid)


class TestLoadDataStrictMode(unittest.TestCase):
    def test_strict_mode_raises_on_invalid_data(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0))
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0))

            with self.assertRaises(ValueError):
                load_data(root, strict=True)

    def test_non_strict_mode_only_warns(self):
        with tempfile.TemporaryDirectory() as root:
            _write_doe(root, "cobyla", "Sphere_2d", "instance_0", _default_rows(f0_0=13.0))
            _write_doe(root, "cobyqa", "Sphere_2d", "instance_0", _default_rows(f0_0=999.0))

            with self.assertWarns(UserWarning):
                data, report = load_data(root, strict=False)

        self.assertFalse(report.is_valid)


if __name__ == "__main__":
    unittest.main()
