import unittest

import numpy as np

from smtoptim.core import Problem
from smtoptim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver

from smtoptim.surrogate_models.smt import SmtAutoModel

from smtoptim.acquisition_strategies import MFSEGO


def rosenbrock(x: np.ndarray) -> np.ndarray:

    ndim = x.ndim

    if ndim == 1:
        x = x.reshape(1, -1)

    n = x.shape[1]
    A = 10

    def temp(x):
        return x**2 - A*np.cos(2*np.pi*x)

    temp_vec = np.vectorize(temp)
    value = (1 - x[:, 0])**2 + 100*(x[:, 1] - x[:, 0]**2)**2

    if ndim == 1:
        value = value.item()

    return value


def disk(x: np.ndarray) -> np.ndarray:

    ndim = x.ndim

    if ndim == 1:
        x = x.reshape(1, -1)

    value = x[:, 0]**2 + x[:, 1]**2 - 1

    if ndim == 1:
        value = value.item()

    return value


class TestOptimization(unittest.TestCase):

    def test_sego_2d_1c(self):

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])

        obj_config = ObjectiveConfig(
            objective=[rosenbrock],
            surrogate=SmtAutoModel,
        )

        cstr_config = ConstraintConfig(
            constraint=[disk],
            type="less",
            value=0.0,
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            cstr_configs=[cstr_config],
            design_space=bounds,
        )

        opt_config = DriverConfig(
            max_iter=10,
            seed=42,
        )

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=MFSEGO,
        )

        state = optimizer.optimize()

        y_data = np.empty(len(state.dataset.samples))
        c_data = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            y_data[i] = sample.obj[0]
            c_data[i] = sample.cstr[0]

        feasible = c_data <= 0

        bo_fmin = y_data[feasible].min()

        self.assertLessEqual(bo_fmin, 0.5)


    # def test_sego_2d_2c(self):
    #     pass


if __name__ == "__main__":
    unittest.main()