import unittest
import numpy as np

from smt_optim.benchmarks.multi_obj import BNH, TNK, OSY, DTLZ5

class TestMOBenchmarks(unittest.TestCase):
    def test_bnh(self):
        prob = BNH()
        x = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        f1 = prob.f1(x)
        f2 = prob.f2(x)
        g1 = prob.g1(x)
        g2 = prob.g2(x)

        np.testing.assert_allclose(f1, [0.0, 8.0])
        np.testing.assert_allclose(f2, [50.0, 32.0])
        np.testing.assert_allclose(g1, [0.0, -8.0])
        np.testing.assert_allclose(g2, [-65.3, -57.3])

    def test_tnk(self):
        prob = TNK()
        x = np.array([
            [1.0, 1.0]
        ])
        f1 = prob.f1(x)
        f2 = prob.f2(x)
        g1 = prob.g1(x)
        g2 = prob.g2(x)

        np.testing.assert_allclose(f1, [1.0])
        np.testing.assert_allclose(f2, [1.0])
        expected_g1 = -1.0 - 1.0 + 1.0 + 0.1 * np.cos(16.0 * np.arctan(1.0))
        np.testing.assert_allclose(g1, [expected_g1])
        np.testing.assert_allclose(g2, [0.0])

    def test_osy(self):
        prob = OSY()
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        f1 = prob.f1(x)
        f2 = prob.f2(x)
        g1 = prob.g1(x)
        g2 = prob.g2(x)
        g3 = prob.g3(x)
        g4 = prob.g4(x)
        g5 = prob.g5(x)
        g6 = prob.g6(x)

        # f1: -(25*(x0-2)^2 + (x1-2)^2 + (x2-1)^2 + (x3-4)^2 + (x4-1)^2)
        # for x=0: -(25*4 + 4 + 1 + 16 + 1) = -122
        # for x=1: -(25*1 + 1 + 0 + 9 + 0) = -35
        np.testing.assert_allclose(f1, [-122.0, -35.0])
        np.testing.assert_allclose(f2, [0.0, 6.0])
        np.testing.assert_allclose(g1, [2.0, 0.0])
        np.testing.assert_allclose(g2, [-6.0, -4.0])
        np.testing.assert_allclose(g3, [-2.0, -2.0])
        np.testing.assert_allclose(g4, [-2.0, -4.0])
        np.testing.assert_allclose(g5, [5.0, 1.0])
        np.testing.assert_allclose(g6, [-5.0, -1.0])

    def test_dtlz5(self):
        prob = DTLZ5()
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        f1 = prob.f1(x)
        f2 = prob.f2(x)
        g = prob.g(x)
        f1_lf = prob.f1_lf(x)
        f2_lf = prob.f2_lf(x)
        g_lf = prob.g_lf(x)

        # For x=0, u(xq) = sum((0-0.5)^2) = 7 * 0.25 = 1.75
        # f1 = (1 + 1.75) * cos(0) = 2.75
        # f2 = (1 + 1.75) * sin(0) = 0.0
        np.testing.assert_allclose(f1, [2.75])
        np.testing.assert_allclose(f2, [0.0])
        np.testing.assert_allclose(g, [2.25])
        
        # f1_lf = (1 + 0.8 * 1.75) * cos(0) = 1 + 1.4 = 2.4
        np.testing.assert_allclose(f1_lf, [2.4])
        np.testing.assert_allclose(f2_lf, [0.0])
        np.testing.assert_allclose(g_lf, [2.4])

if __name__ == "__main__":
    unittest.main()
