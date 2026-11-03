import unittest
from unittest import mock

import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, DriverConfig, Driver

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_functions import init_ivr, init_mean, negate
from smt_optim.acquisition_strategies import qPGP
from smt_optim.acquisition_strategies.qpgp import select_batch


def branin(x: np.ndarray) -> float:
    x = np.atleast_2d(x)
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    value = (
        a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
        + s * (1 - t) * np.cos(x[:, 0])
        + s
    )
    return value.item()


class TestqPGP(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    def run_driver(
        self, batch_size, max_iter, acq_init_1=init_mean, acq_init_2=negate(init_ivr)
    ):
        obj_config = ObjectiveConfig(objective=[branin], surrogate=SmtAutoModel)

        problem = Problem(
            obj_configs=[obj_config],
            design_space=self.bounds,
        )

        driver = Driver(
            problem=problem,
            config=DriverConfig(max_iter=max_iter, nt_init=6, seed=42),
            strategy=qPGP,
            strategy_kwargs=dict(
                acq_init_1=acq_init_1,
                acq_init_2=acq_init_2,
                batch_size=batch_size,
                pop_size=20,
                n_gen=10,
            ),
        )

        return driver.optimize()

    def test_qpgp_batch_size(self):
        batch_size = 3
        max_iter = 2

        state = self.run_driver(batch_size, max_iter)

        # each iteration adds batch_size samples
        self.assertEqual(len(state.dataset.samples), 6 + batch_size * max_iter)

    def test_qpgp_branin(self):
        state = self.run_driver(batch_size=3, max_iter=10)

        f_min = min(sample.obj[0] for sample in state.dataset.samples)

        # global minimum of Branin: 0.398
        self.assertLessEqual(f_min, 1.0)

    def test_qpgp_custom_acquisition_functions(self):
        calls = []

        def init_neg_variance(state):
            model = state.obj_models[0]

            def neg_variance(x):
                calls.append(x.shape)
                return -model.predict_variances(x)

            return neg_variance

        batch_size = 3
        max_iter = 2

        state = self.run_driver(batch_size, max_iter, acq_init_2=init_neg_variance)

        self.assertEqual(len(state.dataset.samples), 6 + batch_size * max_iter)

        # the given acquisition function is evaluated on the whole population
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0], (20, 2))

    def test_select_batch(self):
        rng = np.random.default_rng(0)
        pareto_x = rng.random((30, 2))

        batch = select_batch(pareto_x, 5, seed=0)

        # batch_size distinct points from the Pareto set
        self.assertEqual(batch.shape, (5, 2))
        self.assertEqual(np.unique(batch, axis=0).shape[0], 5)
        for x in batch:
            self.assertTrue(np.any(np.all(np.isclose(pareto_x, x), axis=1)))

        # batch_size = 1 -> one point from the Pareto set
        batch = select_batch(pareto_x, 1, seed=0)
        self.assertEqual(batch.shape, (1, 2))
        self.assertTrue(np.any(np.all(np.isclose(pareto_x, batch[0]), axis=1)))

        # fewer Pareto points than batch_size -> return all of them
        batch = select_batch(pareto_x[:3], 5)
        self.assertEqual(batch.shape, (3, 2))

        # duplicates are not selected twice
        dup_x = np.vstack([pareto_x[:6], pareto_x[:6]])
        batch = select_batch(dup_x, 5, seed=0)
        self.assertEqual(np.unique(batch, axis=0).shape[0], 5)

    def test_select_batch_empty_cluster(self):
        # all the points in a single cluster -> batch completed with the farthest points
        pareto_x = np.array([[0.0, 0.0], [0.1, 0.0], [0.5, 0.0], [1.0, 0.0]])

        with mock.patch(
            "smt_optim.acquisition_strategies.qpgp.kmeans2",
            return_value=(np.array([[0.0, 0.0], [9.0, 9.0]]), np.zeros(4, dtype=int)),
        ):
            batch = select_batch(pareto_x, 2)

        np.testing.assert_array_equal(batch, np.array([[0.0, 0.0], [1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
