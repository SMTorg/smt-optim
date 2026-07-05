import unittest
import numpy as np

from smt_optim.benchmarks.multi_obj import ZDT1, ZDT2, ZDT3, ZDT4


class TestZDTBenchmarks(unittest.TestCase):
    def test_zdt1(self):
        zdt = ZDT1()
        zdt.set_dim(5)
        # Vectorized test (2D)
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
        # 1D test
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(zdt.f1(x2d), [0.0, 1.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = [1.0, 10.0 * (1.0 - np.sqrt(0.1))]
        np.testing.assert_allclose(zdt.f2(x2d), expected_f2)
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2[0])

        # Test individual functions for 1D coverage
        np.testing.assert_allclose(zdt.g(x1d), 1.0)
        np.testing.assert_allclose(zdt.g(x2d), [1.0, 10.0])

    def test_zdt2(self):
        zdt = ZDT2()
        zdt.set_dim(5)
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(zdt.f1(x2d), [0.0, 1.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = [1.0, 9.9]
        np.testing.assert_allclose(zdt.f2(x2d), expected_f2)
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2[0])

        np.testing.assert_allclose(zdt.g(x1d), 1.0)
        np.testing.assert_allclose(zdt.g(x2d), [1.0, 10.0])

    def test_zdt3(self):
        zdt = ZDT3()
        zdt.set_dim(5)
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(zdt.f1(x2d), [0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = [1.0]
        np.testing.assert_allclose(zdt.f2(x2d), expected_f2)
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2[0])

        np.testing.assert_allclose(zdt.g(x1d), 1.0)
        np.testing.assert_allclose(zdt.g(x2d), [1.0])

    def test_zdt4(self):
        zdt = ZDT4()
        zdt.set_dim(10)
        x2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(zdt.f1(x2d), [0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = [1.0]
        np.testing.assert_allclose(zdt.f2(x2d), expected_f2)
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2[0])

        np.testing.assert_allclose(zdt.g(x1d), 1.0)
        np.testing.assert_allclose(zdt.g(x2d), [1.0])


if __name__ == "__main__":
    unittest.main()
