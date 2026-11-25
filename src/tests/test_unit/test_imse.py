import unittest
import warnings

import numpy as np
from scipy import stats
from smt.surrogate_models import KRG
from smt_optim.acquisition_functions.integrated_variance_reduction import (
    current_variance,
    integrated_variance_reduction,
    init_ivr,
    variance_update,
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

    def test_vectorized_integrated_variance_reduction(self):
        # the vectorized rank-1 update gives the same values as one update per candidate
        x_cands = np.array([[0.1], [0.25], [0.6], [0.9], [0.5]])

        imse_current = np.mean(current_variance(self.sm, self.x_mc))
        expected = np.array(
            [
                [
                    imse_current
                    - np.mean(variance_update(self.sm, x[None, :], self.x_mc))
                ]
                for x in x_cands
            ]
        )

        ivr_vals = integrated_variance_reduction(self.sm, x_cands, self.x_mc)
        np.testing.assert_allclose(ivr_vals, expected, atol=1e-12)

        # a training point barely reduces the variance (the model is fitted with noise)
        self.assertAlmostEqual(ivr_vals[-1, 0], 0.0, places=6)

    def test_full_inversion(self):
        # without noise, the full inversion gives the same values as the rank-1 update
        rng = np.random.default_rng(0)
        xt = rng.random((40, 5))
        sm = KRG(print_global=False, nugget=1e-8)
        sm.set_training_values(xt, np.sin(3 * xt).sum(axis=1))
        sm.train()

        x_mc = rng.random((64, 5))
        x_cands = rng.random((5, 5))

        np.testing.assert_allclose(
            integrated_variance_reduction(sm, x_cands, x_mc, inv_block=False),
            integrated_variance_reduction(sm, x_cands, x_mc, inv_block=True),
            atol=1e-10,
        )

    def test_default_integration_points(self):
        x_cands = np.array([[0.25], [0.75]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imse_vals = integrated_variance_reduction(self.sm, points=x_cands)

        self.assertEqual(imse_vals.shape, (2, 1))
        self.assertTrue(np.all(imse_vals >= 0.0))

    def test_unsupported_model(self):
        class NotKriging:
            pass

        with self.assertRaises(TypeError):
            integrated_variance_reduction(NotKriging(), np.array([[0.5]]), self.x_mc)

    def test_init_ivr_invalid_n_integration(self):
        state = DummyState(self.sm, num_dim=1, iter=3)

        with self.assertRaises(ValueError):
            init_ivr(state, n_integration=0)


if __name__ == "__main__":
    unittest.main()
