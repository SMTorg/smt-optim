import unittest
import numpy as np

from smt_optim.benchmarks.multiobj.constrained import BNH, TNK, OSY


class TestMOBenchmarks(unittest.TestCase):
    def test_bnh(self):
        prob = BNH()
        x1d = np.array([0.0, 0.0])
        np.testing.assert_allclose(prob.f1(x1d), 0.0)
        np.testing.assert_allclose(prob.f2(x1d), 50.0)
        np.testing.assert_allclose(prob.g1(x1d), 0.0)
        np.testing.assert_allclose(prob.g2(x1d), -65.3)

    def test_tnk(self):
        prob = TNK()
        x1d = np.array([1.0, 1.0])
        np.testing.assert_allclose(prob.f1(x1d), 1.0)
        np.testing.assert_allclose(prob.f2(x1d), 1.0)

        expected_g1 = -1.0 - 1.0 + 1.0 + 0.1 * np.cos(16.0 * np.arctan(1.0))
        np.testing.assert_allclose(prob.g1(x1d), expected_g1)
        np.testing.assert_allclose(prob.g2(x1d), 0.0)

    def test_osy(self):
        prob = OSY()
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(prob.f1(x1d), -122.0)
        np.testing.assert_allclose(prob.f2(x1d), 0.0)
        np.testing.assert_allclose(prob.g1(x1d), 2.0)
        np.testing.assert_allclose(prob.g2(x1d), -6.0)
        np.testing.assert_allclose(prob.g3(x1d), -2.0)
        np.testing.assert_allclose(prob.g4(x1d), -2.0)
        np.testing.assert_allclose(prob.g5(x1d), 5.0)
        np.testing.assert_allclose(prob.g6(x1d), -5.0)


if __name__ == "__main__":
    unittest.main()
