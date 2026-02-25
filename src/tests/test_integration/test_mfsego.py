import unittest

import numpy as np

from smtoptim.core import Problem
from smtoptim.core import ObjectiveConfig, ConstraintConfig, OptimizerConfig, Optimizer

from smtoptim.surrogate_models.smt import SmtAutoModel

from smtoptim.acquisition_strategies import MFSEGO


def branin_forrester(x):
    X1 = 15 * x[0] - 5
    X2 = 15 * x[1]

    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    f = (a * (X2 - b * X1 ** 2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x[0]

    return f


class Branin1:

    def __init__(self):

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.bounds = np.array([
            [0, 1],
            [0, 1]
        ])

        self.costs = [0.1, 1]

        self.objective = [self.lf_objective, self.hf_objective]
        self.constraints = [
            [self.lf_constraint, self.hf_constraint]
        ]

        # f_min = 5.5757
        # x_min = np.array([0.9677, 0.2067])

    def hf_objective(self, x):
        return branin_forrester(x)

    def lf_objective(self, x):
        return self.hf_objective(x) - np.cos(0.5*x[0]) - x[1]**3

    def hf_constraint(self, x):
        return -x[0]*x[1] + 0.2

    def lf_constraint(self, x):
        return -x[0]*x[1] - 0.7*x[1] + 0.3*x[0]


class TestOptimization(unittest.TestCase):

    def test_mfsego_2d_1c(self):

        max_iter = 3

        branin = Branin1()

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])

        obj_config = ObjectiveConfig(
            objective=branin.objective,
            design_space=branin.bounds,
            surrogate=SmtAutoModel,
            costs=[5, 1],
        )

        cstr_config = ConstraintConfig(
            constraint=branin.constraints[0],
            type="less",
            value=0.0,
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            cstr_configs=[cstr_config],
        )

        opt_config = OptimizerConfig(
            max_iter=max_iter,
            seed=42,
        )

        optimizer = Optimizer(
            problem=problem,
            config=opt_config,
            strategy=MFSEGO,
        )

        state = optimizer.optimize()

        self.assertEqual(state.iter, max_iter)



    def test_fidelity_criteria(self):

        fidelity_criteria = ["optimistic", "pessimistic", "average"]


        max_iter = 1

        branin = Branin1()

        bounds = np.array([
            [-1.5, 1.5],
            [-1.5, 1.5],
        ])


        for fid_crit in fidelity_criteria:
            obj_config = ObjectiveConfig(
                objective=branin.objective,
                design_space=branin.bounds,
                surrogate=SmtAutoModel,
                costs=[5, 1],
            )

            cstr_config = ConstraintConfig(
                constraint=branin.constraints[0],
                type="less",
                value=0.0,
                surrogate=SmtAutoModel,
            )

            problem = Problem(
                obj_configs=[obj_config],
                cstr_configs=[cstr_config],
            )

            opt_config = OptimizerConfig(
                max_iter=max_iter,
                seed=42,
            )

            optimizer = Optimizer(
                problem=problem,
                config=opt_config,
                strategy=MFSEGO,
                strategy_kwargs={
                    "fidelity_crit": fid_crit,
                }
            )

            state = optimizer.optimize()

            self.assertEqual(state.iter, max_iter)


if __name__ == '__main__':
    unittest.main()