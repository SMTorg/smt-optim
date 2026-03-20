import unittest

import numpy as np
import scipy.stats as stats

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver

from smt_optim.surrogate_models.smt import SmtAutoModel

def rosenbrock(x: np.ndarray) -> np.ndarray:

    ndim = x.ndim

    if ndim == 1:
        x = x.reshape(1, -1)

    n = x.shape[1]
    A = 10

    value = (1 - x[:, 0])**2 + 100*(x[:, 1] - x[:, 0]**2)**2

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

        # check hyper_opt


if __name__ == "__main__":
    unittest.main()

