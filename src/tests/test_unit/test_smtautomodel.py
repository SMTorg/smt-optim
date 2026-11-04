import unittest

import numpy as np
import scipy.stats as stats


from smt_optim.surrogate_models.smt import SmtAutoModel


def rosenbrock(x: np.ndarray) -> np.ndarray:

    ndim = x.ndim

    if ndim == 1:
        x = x.reshape(1, -1)

    x.shape[1]

    value = (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2

    if ndim == 1:
        value = value.item()

    return value


class TestSmtAutoModel(unittest.TestCase):
    def test_seed(self):

        sampler = stats.qmc.LatinHypercube(d=2, seed=42)
        xt = sampler.random(10)

        yt = rosenbrock(xt)

        model = SmtAutoModel()
        model.train([xt], [yt.reshape(-1, 1)])

    def test_predict_covariances(self):

        sampler = stats.qmc.LatinHypercube(d=2, seed=42)
        xt = sampler.random(10)

        yt = rosenbrock(xt)

        model = SmtAutoModel()
        model.train([xt], [yt.reshape(-1, 1)])

        x = stats.qmc.LatinHypercube(d=2, seed=0).random(4)
        cov = model.predict_covariances(x)

        self.assertEqual(cov.shape, (4, 4))
        np.testing.assert_allclose(cov, cov.T)

        # the diagonal is the pointwise variance
        np.testing.assert_allclose(
            np.diag(cov).reshape(-1, 1), model.predict_variances(x)
        )


if __name__ == "__main__":
    unittest.main()
