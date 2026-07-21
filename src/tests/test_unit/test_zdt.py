import unittest
import numpy as np

from smt_optim.benchmarks.multiobj.zdt import ZDT1, ZDT2, ZDT3, ZDT4


class TestZDTBenchmarks(unittest.TestCase):
    def test_zdt1(self):
        zdt = ZDT1()
        zdt.set_dim(5)
        # Vectorized test (2D)
        # 1D test
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = 1.0
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2)

        # Test individual functions for 1D coverage
        np.testing.assert_allclose(zdt.g(x1d), 1.0)

    def test_zdt2(self):
        zdt = ZDT2()
        zdt.set_dim(5)
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = 1.0
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2)

        np.testing.assert_allclose(zdt.g(x1d), 1.0)

    def test_zdt3(self):
        zdt = ZDT3()
        zdt.set_dim(5)
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = 1.0
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2)

        np.testing.assert_allclose(zdt.g(x1d), 1.0)

    def test_zdt4(self):
        zdt = ZDT4()
        zdt.set_dim(10)
        x1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(zdt.f1(x1d), 0.0)

        expected_f2 = 1.0
        np.testing.assert_allclose(zdt.f2(x1d), expected_f2)

        np.testing.assert_allclose(zdt.g(x1d), 1.0)


if __name__ == "__main__":
    unittest.main()
