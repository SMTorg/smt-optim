import unittest
import numpy as np

from smt_optim.acquisition_functions import expected_improvement, vec_expected_improvement


class TestExpectedImprovement(unittest.TestCase):

    def test_expected_improvement(self):

        mu = 1.0
        s2 = 1.0
        f_min = 0.5
        print(expected_improvement(mu, s2, f_min))

        mu = 1.0
        s2 = 0.0
        f_min = 0.5
        self.assertEqual(expected_improvement(mu, s2, f_min), 0.0)

    def test_vec_expected_improvement(self):

        mu = np.array([[1.0],
                      [1.0]])
        s2 = np.array([[1.0],
                      [0.0]])
        f_min = 0.5

        vec_ei = vec_expected_improvement(mu, s2, f_min)

        # test output shape
        self.assertEqual(vec_ei.shape[0], 2)
        self.assertEqual(vec_ei.shape[1], 1)

        # test that vec_ei corresponds to ei
        self.assertEqual(vec_ei[0], expected_improvement(mu[0, 0], s2[0, 0], f_min))
        self.assertEqual(vec_ei[1], 0)


if __name__ == "__main__":
    unittest.main()