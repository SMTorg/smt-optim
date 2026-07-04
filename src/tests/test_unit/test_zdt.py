import unittest
import numpy as np

from smt_optim.benchmarks.multi_obj import ZDT1, ZDT2, ZDT3, ZDT4

class TestZDTBenchmarks(unittest.TestCase):
    def test_zdt1(self):
        zdt = ZDT1()
        zdt.set_dim(5)
        # Vectorized test
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        
        f1 = zdt.f1(x)
        f2 = zdt.f2(x)
        
        np.testing.assert_allclose(f1, [0.0, 1.0])
        # For [0,0,0,0,0]: g = 1, f1 = 0, f2 = 1 * (1 - 0) = 1.0
        # For [1,1,1,1,1]: g = 1 + 9*(4/4) = 10, f1 = 1, f2 = 10 * (1 - sqrt(1/10))
        expected_f2 = [1.0, 10.0 * (1.0 - np.sqrt(0.1))]
        np.testing.assert_allclose(f2, expected_f2)

    def test_zdt2(self):
        zdt = ZDT2()
        zdt.set_dim(5)
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        
        f1 = zdt.f1(x)
        f2 = zdt.f2(x)
        
        np.testing.assert_allclose(f1, [0.0, 1.0])
        # For [0,0,0,0,0]: g = 1, f1 = 0, f2 = 1 * (1 - 0) = 1.0
        # For [1,1,1,1,1]: g = 10, f1 = 1, f2 = 10 * (1 - (1/10)^2) = 10 * 0.99 = 9.9
        expected_f2 = [1.0, 9.9]
        np.testing.assert_allclose(f2, expected_f2)

    def test_zdt3(self):
        zdt = ZDT3()
        zdt.set_dim(5)
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        f1 = zdt.f1(x)
        f2 = zdt.f2(x)
        
        np.testing.assert_allclose(f1, [0.0])
        # For [0,0,0,0,0]: g = 1, f1 = 0, f2 = 1 * (1 - 0 - 0) = 1.0
        expected_f2 = [1.0]
        np.testing.assert_allclose(f2, expected_f2)

    def test_zdt4(self):
        zdt = ZDT4()
        zdt.set_dim(10)
        x = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        f1 = zdt.f1(x)
        f2 = zdt.f2(x)
        
        np.testing.assert_allclose(f1, [0.0])
        # For x=0: g = 1 + 10*(9) + sum(-10) = 1 + 90 - 90 = 1.0
        # f2 = 1.0 * (1.0 - 0) = 1.0
        expected_f2 = [1.0]
        np.testing.assert_allclose(f2, expected_f2)

if __name__ == "__main__":
    unittest.main()
