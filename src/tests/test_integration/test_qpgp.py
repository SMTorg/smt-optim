import unittest
from unittest import mock

import numpy as np
import smt.design_space as ds

from smt_optim.core import Problem
from smt_optim.core import ConstraintConfig, ObjectiveConfig, DriverConfig, Driver
from smt_optim.core.state import State

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

    def test_qpgp_reproducibility(self):
        state_1 = self.run_driver(batch_size=3, max_iter=2)
        state_2 = self.run_driver(batch_size=3, max_iter=2)

        # two runs with the same seed give the same points and values
        for attr in ["x", "obj"]:
            values_1 = np.array([getattr(s, attr) for s in state_1.dataset.samples])
            values_2 = np.array([getattr(s, attr) for s in state_2.dataset.samples])
            np.testing.assert_array_equal(values_1, values_2)

    def test_qpgp_validate_config(self):
        def build(design_space=None, num_obj=1, num_fidelity=1, constrained=False):
            problem = Problem(
                obj_configs=[
                    ObjectiveConfig(
                        objective=[branin] * num_fidelity, surrogate=SmtAutoModel
                    )
                    for _ in range(num_obj)
                ],
                design_space=self.bounds if design_space is None else design_space,
                cstr_configs=(
                    [ConstraintConfig([branin], upper=0.0, surrogate=SmtAutoModel)]
                    if constrained
                    else []
                ),
                costs=[1.0] * num_fidelity,
            )
            return State(problem)

        acq_inits = dict(acq_init_1=init_mean, acq_init_2=negate(init_ivr))
        mixed = ds.DesignSpace(
            [ds.FloatVariable(-5.0, 10.0), ds.IntegerVariable(0, 15)]
        )

        invalid = [
            (build(num_obj=2), acq_inits, ValueError),
            (build(constrained=True), acq_inits, ValueError),
            (build(num_fidelity=2), acq_inits, ValueError),
            (build(design_space=mixed), acq_inits, ValueError),
            (build(), dict(**acq_inits, batch_size=0), ValueError),
            (build(), dict(acq_init_1=None, acq_init_2=init_mean), TypeError),
        ]

        for state, kwargs, error in invalid:
            with self.assertRaises(error):
                qPGP(state, **kwargs)

        # valid configuration
        qPGP(build(), **acq_inits)

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

        # batch_size < 1
        with self.assertRaises(ValueError):
            select_batch(pareto_x, 0)

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
