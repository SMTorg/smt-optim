import unittest

import numpy as np

from smtoptim.core import Problem
from smtoptim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver

from smtoptim.surrogate_models.smt import SmtAutoModel

from smtoptim.acquisition_strategies import MFSEGO


def func_1d(x):
    return ((x - 3.5) * np.sin((x - 3.5) / (np.pi))).item()


def func_2d(x):
    pass


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


class TestOptimization(unittest.TestCase):

    def test_ego_1d(self):

        bounds = np.array([[0, 25]])

        obj_config = ObjectiveConfig(
            objective=[func_1d],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=bounds,
            costs=[1],
        )

        opt_config = DriverConfig(
            max_iter=5,
            seed=42,
        )


        optimizer = Driver(problem, config=opt_config, strategy=MFSEGO)
        state = optimizer.optimize()

        y_data = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            y_data[i] = sample.obj[0]
        bo_fmin = y_data.min()

        self.assertLessEqual (bo_fmin, -14.)


    def test_ego_2d(self):

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])

        obj_config = ObjectiveConfig(
            objective=[rosenbrock],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=bounds,
            cstr_configs=[],
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
        for i, sample in enumerate(state.dataset.samples):
            y_data[i] = sample.obj[0]
        bo_fmin = y_data.min()

        self.assertLessEqual(bo_fmin, 1.)


if __name__ == "__main__":
    unittest.main()