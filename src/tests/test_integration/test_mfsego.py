import unittest

import numpy as np

from smt.applications import MFK

from smt_optim.core import Problem
from smt_optim.core import (
    ObjectiveConfig,
    ConstraintConfig,
    DriverConfig,
    Driver,
    State,
)

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import SEGO
from smt_optim.acquisition_strategies.sego import compute_sigma2_red


def branin_forrester(x):
    X1 = 15 * x[0] - 5
    X2 = 15 * x[1]

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    f = (
        a * (X2 - b * X1**2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e
    ) + 5 * x[0]

    return f


class Branin1:
    def __init__(self):

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.bounds = np.array([[0, 1], [0, 1]])

        self.costs = [0.1, 1]

        self.objective = [self.lf_objective, self.hf_objective]
        self.constraints = [[self.lf_constraint, self.hf_constraint]]

        # f_min = 5.5757
        # x_min = np.array([0.9677, 0.2067])

    def hf_objective(self, x):
        return branin_forrester(x)

    def lf_objective(self, x):
        return self.hf_objective(x) - np.cos(0.5 * x[0]) - x[1] ** 3

    def hf_constraint(self, x):
        return -x[0] * x[1] + 0.2

    def lf_constraint(self, x):
        return -x[0] * x[1] - 0.7 * x[1] + 0.3 * x[0]


def sasena2002_hf(x):
    return -np.sin(x) - np.exp(x / 100) + 10


def sasena2002_lf(x):
    return sasena2002_hf(x) + 0.3 + 0.03 * (x - 3) ** 2


class TestOptimization(unittest.TestCase):
    def test_mfsego_2d_1c(self):

        max_iter = 3

        branin = Branin1()

        np.array(
            [
                [-1.5, 1.5],
                [-1.5, 1.5],
            ]
        )

        obj_config = ObjectiveConfig(
            objective=branin.objective,
            surrogate=SmtAutoModel,
        )

        cstr_config = ConstraintConfig(
            constraint=branin.constraints[0],
            upper=0.0,
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=branin.bounds,
            cstr_configs=[cstr_config],
            costs=[5, 1],
        )

        opt_config = DriverConfig(
            max_iter=max_iter,
            seed=42,
        )

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=SEGO,
        )

        state = optimizer.optimize()

        self.assertEqual(state.iter, max_iter)

    def test_fidelity_criteria(self):

        fidelity_criteria = ["optimistic", "pessimistic", "average"]

        max_iter = 1

        branin = Branin1()

        np.array(
            [
                [-1.5, 1.5],
                [-1.5, 1.5],
            ]
        )

        for fid_crit in fidelity_criteria:
            obj_config = ObjectiveConfig(
                objective=branin.objective,
                surrogate=SmtAutoModel,
            )

            cstr_config = ConstraintConfig(
                constraint=branin.constraints[0],
                upper=0.0,
                surrogate=SmtAutoModel,
            )

            problem = Problem(
                obj_configs=[obj_config],
                design_space=branin.bounds,
                cstr_configs=[cstr_config],
                costs=[5, 1],
            )

            opt_config = DriverConfig(
                max_iter=max_iter,
                seed=42,
            )

            optimizer = Driver(
                problem=problem,
                config=opt_config,
                strategy=SEGO,
                strategy_kwargs={
                    "fidelity_crit": fid_crit,
                },
            )

            state = optimizer.optimize()

            self.assertEqual(state.iter, max_iter)

    def test_get_fidelity(self):

        x_train = [np.linspace(0, 10, 6), np.array([2, 4, 6])]

        y_train = [sasena2002_lf(x_train[0]), sasena2002_hf(x_train[1])]

        model = MFK(print_global=False, n_start=3, hyper_opt="Cobyla", seed=42)
        model.set_training_values(
            x_train[0].reshape(-1, 1), y_train[0].reshape(-1, 1), name=0
        )
        model.set_training_values(x_train[1].reshape(-1, 1), y_train[1].reshape(-1, 1))
        model.train()

        obj_config = ObjectiveConfig(
            objective=[sasena2002_lf, sasena2002_hf],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=np.array([[0, 10]]),
            costs=[0.2, 1],
        )

        state = State(problem)
        state.obj_models[0].model = model

        mfsego = SEGO(state)

        next_x = np.array([[5]])
        level = mfsego.get_fidelity(next_x, state)[0]
        self.assertEqual(level, 0)

        next_x = np.array([[8]])
        level = mfsego.get_fidelity(next_x, state)[0]
        self.assertEqual(level, 1)

    def test_get_fidelity_multiple_points(self):
        # LF samples: 0, 2, 4, 6, 8, 10 / HF samples: 2, 4, 6
        x_train = [np.linspace(0, 10, 6), np.array([2, 4, 6])]

        y_train = [sasena2002_lf(x_train[0]), sasena2002_hf(x_train[1])]

        model = MFK(print_global=False, n_start=3, hyper_opt="Cobyla", seed=42)
        model.set_training_values(
            x_train[0].reshape(-1, 1), y_train[0].reshape(-1, 1), name=0
        )
        model.set_training_values(x_train[1].reshape(-1, 1), y_train[1].reshape(-1, 1))
        model.train()

        obj_config = ObjectiveConfig(
            objective=[sasena2002_lf, sasena2002_hf],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=np.array([[0, 10]]),
            costs=[0.2, 1],
        )

        state = State(problem)
        state.obj_models[0].model = model

        sego = SEGO(state)

        # points are interleaved so that a mix-up between rows would be detected
        lf_only_points = (0.0, 8.0, 10.0)  # LF sample but no HF sample
        far_points = (1.0, 3.0, 5.0, 7.0, 9.0)  # midway between LF samples
        next_x = np.array([0, 1, 8, 3, 10, 5, 9, 7], dtype=float).reshape(-1, 1)

        levels = np.asarray(sego.get_fidelity(next_x, state))

        # one fidelity level per query point
        self.assertEqual(levels.shape, (next_x.shape[0],))

        for x, level in zip(next_x.ravel(), levels):
            if x in lf_only_points:
                # an LF sample already exists -> request at least the level above
                self.assertGreaterEqual(level, 1, msg=f"x={x}")
            elif x in far_points:
                # far from any sample -> request the lowest fidelity level
                self.assertEqual(level, 0, msg=f"x={x}")

        # evaluating the points together must be equivalent to evaluating them one by one
        single_levels = [sego.get_fidelity(x.reshape(1, -1), state)[0] for x in next_x]
        self.assertEqual(levels.tolist(), single_levels)

    def test_compute_sigma2_red(self):
        x_train = [np.linspace(0, 10, 6), np.array([2, 4, 6])]

        y_train = [sasena2002_lf(x_train[0]), sasena2002_hf(x_train[1])]

        model = MFK(print_global=False, n_start=3, hyper_opt="Cobyla", seed=42)
        model.set_training_values(
            x_train[0].reshape(-1, 1), y_train[0].reshape(-1, 1), name=0
        )
        model.set_training_values(x_train[1].reshape(-1, 1), y_train[1].reshape(-1, 1))
        model.train()

        obj_config = ObjectiveConfig(
            objective=[sasena2002_lf, sasena2002_hf],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config],
            design_space=np.array([[0, 10]]),
            costs=[0.2, 1],
        )

        state = State(problem)
        state.obj_models[0].model = model

        # mfsego = SEGO(state)

        x_valid = np.linspace(0, 10, 101)

        s2_red = compute_sigma2_red(
            x_valid.reshape(-1, 1), state.obj_models[0], method=None
        )
        delta = s2_red[:, 0] - s2_red[:, 1]

        self.assertTrue(all(d < 0 for d in delta))


if __name__ == "__main__":
    unittest.main()
