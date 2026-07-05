import unittest
import numpy as np

from smt_optim.benchmarks.multi_obj import BNH, TNK, OSY, DTLZ5


class TestMOBenchmarks(unittest.TestCase):
    def test_bnh(self):
        prob = BNH()
        x2d = np.array([[0.0, 0.0], [1.0, 1.0]])
        x1d = np.array([0.0, 0.0])

        np.testing.assert_allclose(prob.f1(x2d), [0.0, 8.0])
        np.testing.assert_allclose(prob.f1(x1d), 0.0)

        np.testing.assert_allclose(prob.f2(x2d), [50.0, 32.0])
        np.testing.assert_allclose(prob.f2(x1d), 50.0)

        np.testing.assert_allclose(prob.g1(x2d), [0.0, -8.0])
        np.testing.assert_allclose(prob.g1(x1d), 0.0)

        np.testing.assert_allclose(prob.g2(x2d), [-65.3, -57.3])
        np.testing.assert_allclose(prob.g2(x1d), -65.3)

    def test_tnk(self):
        prob = TNK()
        x2d = np.array([[1.0, 1.0]])
        x1d = np.array([1.0, 1.0])

        np.testing.assert_allclose(prob.f1(x2d), [1.0])
        np.testing.assert_allclose(prob.f1(x1d), 1.0)

        np.testing.assert_allclose(prob.f2(x2d), [1.0])
        np.testing.assert_allclose(prob.f2(x1d), 1.0)

        expected_g1 = -1.0 - 1.0 + 1.0 + 0.1 * np.cos(16.0 * np.arctan(1.0))
        np.testing.assert_allclose(prob.g1(x2d), [expected_g1])
        np.testing.assert_allclose(prob.g1(x1d), expected_g1)

        np.testing.assert_allclose(prob.g2(x2d), [0.0])
        np.testing.assert_allclose(prob.g2(x1d), 0.0)

    def test_osy(self):
        prob = OSY()
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(prob.f1(x2d), [-122.0, -35.0])
        np.testing.assert_allclose(prob.f1(x1d), -122.0)

        np.testing.assert_allclose(prob.f2(x2d), [0.0, 6.0])
        np.testing.assert_allclose(prob.f2(x1d), 0.0)

        np.testing.assert_allclose(prob.g1(x2d), [2.0, 0.0])
        np.testing.assert_allclose(prob.g1(x1d), 2.0)

        np.testing.assert_allclose(prob.g2(x2d), [-6.0, -4.0])
        np.testing.assert_allclose(prob.g2(x1d), -6.0)

        np.testing.assert_allclose(prob.g3(x2d), [-2.0, -2.0])
        np.testing.assert_allclose(prob.g3(x1d), -2.0)

        np.testing.assert_allclose(prob.g4(x2d), [-2.0, -4.0])
        np.testing.assert_allclose(prob.g4(x1d), -2.0)

        np.testing.assert_allclose(prob.g5(x2d), [5.0, 1.0])
        np.testing.assert_allclose(prob.g5(x1d), 5.0)

        np.testing.assert_allclose(prob.g6(x2d), [-5.0, -1.0])
        np.testing.assert_allclose(prob.g6(x1d), -5.0)

    def test_dtlz5(self):
        prob = DTLZ5()
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(prob.f1(x2d), [2.75])
        np.testing.assert_allclose(prob.f1(x1d), 2.75)

        np.testing.assert_allclose(prob.f2(x2d), [0.0])
        np.testing.assert_allclose(prob.f2(x1d), 0.0)

        np.testing.assert_allclose(prob.g(x2d), [2.25])
        np.testing.assert_allclose(prob.g(x1d), 2.25)

        np.testing.assert_allclose(prob.f1_lf(x2d), [2.4])
        np.testing.assert_allclose(prob.f1_lf(x1d), 2.4)

        np.testing.assert_allclose(prob.f2_lf(x2d), [0.0])
        np.testing.assert_allclose(prob.f2_lf(x1d), 0.0)

        np.testing.assert_allclose(prob.g_lf(x2d), [2.4])
        np.testing.assert_allclose(prob.g_lf(x1d), 2.4)

        np.testing.assert_allclose(prob.u(x1d[3:]), 1.75)
        np.testing.assert_allclose(prob.u(x2d[:, 3:]), [1.75])


if __name__ == "__main__":
    unittest.main()
