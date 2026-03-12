import unittest

import numpy as np

from smtoptim.core import Problem
from smtoptim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver

from smtoptim.surrogate_models.smt import SmtAutoModel

from smtoptim.acquisition_strategies import MFSEGO


def func_1d(x):
    return ((x - 3.5) * np.sin((x - 3.5) / (np.pi))).item()


class TwoConstraints:

    def __init__(self):

        self.num_dim = 2
        self.num_cstr = 2
        self.num_fidelity = 1
        self.bounds = np.array([
            [1, 4],
            [1, 4]
        ])

        self.costs = [1]

        self.objective = [self.func]
        self.constraints = [
            [self.cstr1],
            [self.cstr2]
        ]

    def func(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x[:, 0] ** 2 + x[:, 1] ** 2).item()

    def cstr1(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (0.15*(x[:,0]-4)**2 + 0.1*(x[:,1]-4)**2 -1).item()

    def cstr2(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x[:, 0] - 0.8 * x[:, 1]).item()

class TestOptimization(unittest.TestCase):


    def test_scaling(self):

        two_cstrs = TwoConstraints()

        obj_config = ObjectiveConfig(
            objective=two_cstrs.objective,
            surrogate=SmtAutoModel,
        )

        cstr0_config = ConstraintConfig(
            constraint=two_cstrs.constraints[0],
            type="less",
            value=0.0,
            surrogate=SmtAutoModel,
        )

        cstr1_config = ConstraintConfig(
            constraint=two_cstrs.constraints[1],
            type="less",
            value=0.0,
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=two_cstrs.bounds,
            cstr_configs=[cstr0_config, cstr0_config],
        )

        opt_config = DriverConfig(
            max_iter=1,
            seed=42,
            scaling=True,
        )

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=MFSEGO,
        )

        state = optimizer.optimize()

        # objective scaling
        self.assertAlmostEqual(np.mean(state.scaled_dataset.export_data(0, 0)), 0)
        self.assertAlmostEqual(np.std(state.scaled_dataset.export_data(0, 0)), 1)

        # constraint 1 scaling
        self.assertAlmostEqual(np.std(state.scaled_dataset.export_data(1, 0)), 1)

        # constraints 2 scaling
        self.assertAlmostEqual(np.std(state.scaled_dataset.export_data(2, 0)), 1)

    def test_seed(self):

        bounds = np.array([[0, 25]])

        # --- first optimization ---
        obj_config = ObjectiveConfig(
            objective=[func_1d],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=bounds,
        )

        opt_config = DriverConfig(
            max_iter=1,
            seed=42,
        )

        optimizer = Driver(problem, config=opt_config, strategy=MFSEGO)
        state = optimizer.optimize()

        doe_0 = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            doe_0[i] = sample.x[0]

        # --- second optimization ---
        obj_config = ObjectiveConfig(
            objective=[func_1d],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=bounds,
        )

        opt_config = DriverConfig(
            max_iter=1,
            seed=42,
        )

        optimizer = Driver(problem, config=opt_config, strategy=MFSEGO)
        state = optimizer.optimize()

        doe_1 = np.empty(len(state.dataset.samples))
        for i, sample in enumerate(state.dataset.samples):
            doe_1[i] = sample.x[0]

        # DOE should be identical
        mse = np.mean(doe_0 - doe_1)**2

        self.assertAlmostEqual(mse, 0)


