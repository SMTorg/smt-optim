# @autor Paul-Saves
import unittest
import numpy as np
from smt.applications import MFK
from smt_optim.acquisition_functions.integrated_variance_reduction import (
    integrated_variance_reduction,
)


class TestIMSEIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 1D test problem for MFK
        self.xt_c = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        self.yt_c = np.array([[0.0], [0.5], [1.0], [0.5], [0.0]])

        self.xt_e = np.array([[0.0], [0.5], [1.0]])
        self.yt_e = np.array([[0.0], [1.0], [0.0]])

        self.sm = MFK(theta0=[1e-2], print_global=False)
        self.sm.set_training_values(self.xt_c, self.yt_c, name=0)
        self.sm.set_training_values(self.xt_e, self.yt_e)
        self.sm.train()

        # Integration points
        self.x_mc = np.linspace(0, 1, 50).reshape(-1, 1)

    def test_imse_mfk(self):
        # Candidate point
        x_cand = np.array([[0.5]])

        imse_val = integrated_variance_reduction(
            self.sm, points=x_cand, integration_points=self.x_mc, inv_block=True
        )

        self.assertEqual(imse_val.shape, (1, 1))
        self.assertGreaterEqual(imse_val[0, 0], 0.0)

    def test_vec_imse_mfk(self):
        # Multiple candidate points
        x_cands = np.array([[0.25], [0.75]])

        imse_vals = integrated_variance_reduction(
            self.sm, points=x_cands, integration_points=self.x_mc, inv_block=True
        )

        self.assertEqual(imse_vals.shape, (2, 1))
        self.assertGreaterEqual(imse_vals[0, 0], 0.0)
        self.assertGreaterEqual(imse_vals[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
