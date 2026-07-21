import unittest
import numpy as np

from smt_optim.acquisition_functions.ehvi import ehvi_2o, psi


class TestEHVI(unittest.TestCase):
    def test_psi_basic(self):
        """Test the psi helper function mathematically."""
        a, b = 1.0, 1.0
        mu, s = 0.5, 0.1
        val = psi(a, b, mu, s)
        self.assertGreater(val, 0.0)

    def test_ehvi_2o_basic(self):
        """Test the 2-objective EHVI mathematical function."""
        mu = np.array([1.5, 1.5])
        s = np.array([0.5, 0.5])

        # Augmented Pareto front: must include [-inf, max_f2] and [max_f1, -inf] conceptually
        # But according to the SMT-Optim implementation, Y is the augmented pareto front sorted.
        Y = np.array([[-1e10, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, -1e10]])
        Y = Y[np.argsort(Y[:, 1])]

        val = ehvi_2o(mu, s, Y)
        self.assertGreater(val, 0.0)
        self.assertTrue(np.isfinite(val))

    def test_ehvi_2o_zero_variance_handling(self):
        """
        Test that ehvi_2o handles extremely small variances correctly.
        When s is exactly zero, psi would divide by zero. In practice,
        the acquisition function initializer clips s to sqrt(1e-16) = 1e-8.
        We verify this small value is numerically stable.
        """
        mu = np.array([3.0, 3.0])  # Dominated point
        s = np.array([1e-8, 1e-8])  # Simulated clipped zero variance

        Y = np.array([[-1e10, 2.0], [1.0, 1.0], [2.0, -1e10]])
        Y = Y[np.argsort(Y[:, 1])]

        val = ehvi_2o(mu, s, Y)
        # Should be extremely close to 0 since it's highly dominated and variance is essentially 0
        self.assertAlmostEqual(val, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
