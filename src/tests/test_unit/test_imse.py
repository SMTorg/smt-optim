import unittest

import numpy as np
from scipy import stats
from smt.surrogate_models import KRG
from smt_optim.acquisition_functions.integrated_variance_reduction import (
    integrated_variance_reduction,
    init_ivr,
)


class DummyProblem:
    def __init__(self, num_dim):
        self.num_dim = num_dim


class DummyState:
    def __init__(self, model, num_dim, iter):
        self.obj_models = [model]
        self.problem = DummyProblem(num_dim)
        self.iter = iter


class TestIMSEUnit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 1D test problem
        self.xt = np.array([[0.0], [0.5], [1.0]])
        self.yt = np.array([[0.0], [1.0], [0.0]])

        self.sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8)
        self.sm.set_training_values(self.xt, self.yt)
        self.sm.train()

        # Integration points
        self.x_mc = np.linspace(0, 1, 50).reshape(-1, 1)

    def test_integrated_variance_reduction(self):
        # Candidate point
        x_cand = np.array([[0.25]])

        imse_val = integrated_variance_reduction(
            self.sm, points=x_cand, integration_points=self.x_mc, inv_block=True
        )

        self.assertEqual(imse_val.shape, (1, 1))
        self.assertGreaterEqual(imse_val[0, 0], 0.0)

        # Without block inversion
        imse_val_no_block = integrated_variance_reduction(
            self.sm, points=x_cand, integration_points=self.x_mc, inv_block=False
        )
        self.assertAlmostEqual(imse_val[0, 0], imse_val_no_block[0, 0], places=3)

    def test_vec_integrated_variance_reduction(self):
        # Multiple candidate points
        x_cands = np.array([[0.25], [0.75]])

        imse_vals = integrated_variance_reduction(
            self.sm, points=x_cands, integration_points=self.x_mc, inv_block=True
        )

        self.assertEqual(imse_vals.shape, (2, 1))
        self.assertGreaterEqual(imse_vals[0, 0], 0.0)
        self.assertGreaterEqual(imse_vals[1, 0], 0.0)

    def test_init_ivr(self):
        state = DummyState(self.sm, num_dim=1, iter=3)
        x_cands = np.array([[0.25], [0.75]])

        integration_points = stats.qmc.Sobol(d=1, seed=3).random(8)
        expected = integrated_variance_reduction(self.sm, x_cands, integration_points)

        ivr_vals = init_ivr(state, n_integration=8)(x_cands)
        self.assertEqual(ivr_vals.shape, (2, 1))
        np.testing.assert_allclose(ivr_vals, expected)


if __name__ == "__main__":
    unittest.main()
