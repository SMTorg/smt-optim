import unittest
import numpy as np

from smt_optim.benchmarks.multiobj import zdt_mf
from smt_optim.benchmarks.multiobj import zdt

from pathlib import Path

import smt_optim.benchmarks.multiobj as multiobj_pkg


DATA_DIR = Path(multiobj_pkg.__file__).parent / "data_onera"


class _ReferenceDOEMixin:
    """Compare problem outputs against reference DOE files from ONERA notebook implementations."""

    cls = None
    csv_name = None

    def test_matches_reference_doe(self):
        data = np.genfromtxt(DATA_DIR / self.csv_name, delimiter=",", names=True)
        names = data.dtype.names
        x_cols = [n for n in names if n.startswith("x")]
        X = np.column_stack([data[n] for n in x_cols])

        prob = self.cls()
        prob.set_dim(len(x_cols))

        funcs = {
            "f0_HF": prob.objective[0][-1],
            "f0_LF": prob.objective[0][0],
            "f1_HF": prob.objective[1][-1],
            "f1_LF": prob.objective[1][0],
        }

        for col, fn in funcs.items():
            actual = np.array([fn(x) for x in X])
            with self.subTest(column=col):
                np.testing.assert_allclose(
                    actual, data[col], rtol=1e-9, atol=1e-12, err_msg=col
                )


class TestZDT1Reference(_ReferenceDOEMixin, unittest.TestCase):
    cls = zdt_mf.MF_ZDT1
    csv_name = "ZDT1_6d.csv"


class TestZDT2Reference(_ReferenceDOEMixin, unittest.TestCase):
    cls = zdt_mf.MF_ZDT2
    csv_name = "ZDT2_6d.csv"


class TestZDT3Reference(_ReferenceDOEMixin, unittest.TestCase):
    cls = zdt_mf.MF_ZDT3
    csv_name = "ZDT3_6d.csv"


class TestDTLZ5Reference(_ReferenceDOEMixin, unittest.TestCase):
    cls = zdt_mf.MF_DTLZ5
    csv_name = "DTLZ5_6d.csv"


class _SingleFidelityReferenceDOEMixin:
    """Compare single-fidelity problem outputs against the high-fidelity reference data."""

    cls = None
    csv_name = None

    def test_matches_reference_doe_hf(self):
        data = np.genfromtxt(DATA_DIR / self.csv_name, delimiter=",", names=True)
        x_cols = [n for n in data.dtype.names if n.startswith("x")]
        X = np.column_stack([data[n] for n in x_cols])

        prob = self.cls()
        prob.set_dim(len(x_cols))

        funcs = {"f0_HF": prob.objective[0], "f1_HF": prob.objective[1]}

        for col, fn in funcs.items():
            actual = np.array([fn(x) for x in X])
            with self.subTest(column=col):
                np.testing.assert_allclose(
                    actual, data[col], rtol=1e-9, atol=1e-12, err_msg=col
                )


class TestSingleFidelityZDT1Reference(
    _SingleFidelityReferenceDOEMixin, unittest.TestCase
):
    cls = zdt.ZDT1
    csv_name = "ZDT1_6d.csv"


class TestSingleFidelityZDT2Reference(
    _SingleFidelityReferenceDOEMixin, unittest.TestCase
):
    cls = zdt.ZDT2
    csv_name = "ZDT2_6d.csv"


class TestSingleFidelityZDT3Reference(
    _SingleFidelityReferenceDOEMixin, unittest.TestCase
):
    cls = zdt.ZDT3
    csv_name = "ZDT3_6d.csv"


if __name__ == "__main__":
    unittest.main()
