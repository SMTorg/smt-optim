import unittest

import numpy as np

from smtoptim.core import Problem
from smtoptim.core import ObjectiveConfig, ConstraintConfig, OptimizerConfig, Optimizer

from smtoptim.surrogate_models.smt import SmtSimpleKRG

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


def disk(x: np.ndarray) -> np.ndarray:

    ndim = x.ndim

    if ndim == 1:
        x = x.reshape(1, -1)

    value = x[:, 0]**2 + x[:, 1]**2 - 1

    if ndim == 1:
        value = value.item()

    return value


class TestOptimization(unittest.TestCase):

    def test_ego_1d(self):

        bounds = np.array([[0, 25]])

        obj_config = ObjectiveConfig(
            objective=[func_1d],
            design_space=bounds,
            surrogate=SmtSimpleKRG,
        )

        problem = Problem(
            obj_configs=[obj_config],
        )

        opt_config = OptimizerConfig(
            max_iter=5,
            seed=42,
        )


        optimizer = Optimizer(problem, config=opt_config, strategy=MFSEGO)
        state = optimizer.optimize()

        y_data = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            y_data[i] = sample.obj[0]
        bo_fmin = y_data.min()

        self.assertAlmostEqual (bo_fmin, -15.125, 3)


    def test_ego_2d(self):

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])

        obj_config = ObjectiveConfig(
            objective=[rosenbrock],
            design_space=bounds,
            surrogate=SmtSimpleKRG,
        )

        problem = Problem(
            obj_configs=[obj_config],
            cstr_configs=[]
        )

        opt_config = OptimizerConfig(
            max_iter=10,
            seed=42,
        )

        optimizer = Optimizer(
            problem=problem,
            config=opt_config,
            strategy=MFSEGO,
        )

        state = optimizer.optimize()

        y_data = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            y_data[i] = sample.obj[0]
        bo_fmin = y_data.min()

        # TODO: fix stochastic behavior when seed is provided
        self.assertLessEqual(bo_fmin, 2.)


    def test_sego_2d_1c(self):

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])

        obj_config = ObjectiveConfig(
            objective=[rosenbrock],
            design_space=bounds,
            surrogate=SmtSimpleKRG,
        )

        cstr_config = ConstraintConfig(
            constraint=[disk],
            type="less",
            value=0.0,
            surrogate=SmtSimpleKRG,
        )

        problem = Problem(
            obj_configs=[obj_config],
            cstr_configs=[cstr_config],
        )

        opt_config = OptimizerConfig(
            max_iter=10,
            seed=42,
        )

        optimizer = Optimizer(
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