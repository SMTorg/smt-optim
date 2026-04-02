import unittest

import numpy as np
import numpy.testing as npt

import scipy.stats as stats


from smt_optim.surrogate_models.smt import SmtMFCK

from smt.applications import NestedLHS


class TestSmtMFCK(unittest.TestCase):

    def test_level_covariance(self):
        """
        using .predict_level_covariances should be equivalent to .predict_variances when both level
        parameters are equal.
        """

        bounds = np.array([[0, 1]])

        sampler = NestedLHS(xlimits=bounds, nlevel=2, seed=42)
        xt = sampler(5)
        yt = []
        for lvl in range(2):
            np.random.seed(42+lvl)
            ar = np.random.rand(xt[lvl].shape[0])
            yt.append(ar)

        model = SmtMFCK()
        model.train(xt, yt)

        x_test = np.linspace(0, 1, 51).reshape(-1, 1)

        mu, s2 = model.model.predict_all_levels(x_test)
        cov = []
        for lvl in range(2):
            s2[lvl] = s2[lvl].reshape(-1, 1)
            cov.append(model.predict_level_covariances(x_test, lvl, lvl))
            npt.assert_array_almost_equal(s2[lvl], cov[lvl])


if __name__ == "__main__":
    unittest.main()