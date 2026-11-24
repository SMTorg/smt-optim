import unittest

import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver
from smt_optim.core.state import State

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import qSEGO


def sphere(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(x**2, axis=1)


def sphere_cstr(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(x, axis=1) - 1.0


def make_problem(constrained: bool = False, num_fidelity: int = 1) -> Problem:
    objective = [sphere] * num_fidelity
    obj_config = ObjectiveConfig(objective=objective, surrogate=SmtAutoModel)

    kwargs = {}
    if constrained:
        kwargs["cstr_configs"] = [
            ConstraintConfig(
                constraint=[sphere_cstr] * num_fidelity,
                upper=0.0,
                surrogate=SmtAutoModel,
            )
        ]

    if num_fidelity > 1:
        kwargs["costs"] = [0.2] * (num_fidelity - 1) + [1.0]

    return Problem(
        obj_configs=[obj_config],
        design_space=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        **kwargs,
    )


class RandomQSEGO(qSEGO):
    """qSEGO with a placeholder `get_location` returning random points in the unit hypercube."""

    n_samples_at_call: list[int] = []

    def get_location(self, state: State) -> np.ndarray:
        # records the dataset size each time a batch is requested
        self.n_samples_at_call.append(len(state.dataset.samples))

        rng = np.random.default_rng(state.iter)
        return rng.random((self.batch_size, state.problem.num_dim))


class TestQSEGO(unittest.TestCase):
    def test_get_location_not_implemented(self):
        state = State(make_problem())
        strategy = qSEGO(state)

        with self.assertRaises(NotImplementedError):
            strategy.get_location(state)

    def test_default_parameters(self):
        state = State(make_problem())
        strategy = qSEGO(state)

        self.assertEqual(strategy.batch_size, 3)
        self.assertEqual(strategy.n_samples, 512)
        self.assertEqual(strategy.n_start, 20)
        self.assertEqual(strategy.sp_method, "L-BFGS-B")

    def test_constrained_and_multi_fidelity_allowed(self):
        # accepted by the base class: their support depends on `get_location`
        qSEGO(State(make_problem(constrained=True)))
        qSEGO(State(make_problem(num_fidelity=2)))

    def test_invalid_parameters(self):
        state = State(make_problem())

        with self.assertRaises(ValueError):
            qSEGO(state, batch_size=0)

        with self.assertRaises(ValueError):
            qSEGO(state, n_samples=0)

    def test_batch_pipeline_with_random_points(self):
        """
        Checks that a batch returned by `get_location` goes through `get_infill` and is sampled.
        """

        batch_size = 4
        max_iter = 2

        RandomQSEGO.n_samples_at_call = []

        optimizer = Driver(
            problem=make_problem(),
            config=DriverConfig(max_iter=max_iter, seed=42),
            strategy=RandomQSEGO,
            strategy_kwargs={"batch_size": batch_size},
        )

        state = optimizer.optimize()

        calls = RandomQSEGO.n_samples_at_call
        self.assertEqual(len(calls), max_iter)

        # each iteration adds `batch_size` points (and not 1) to the dataset
        sizes = calls + [len(state.dataset.samples)]
        self.assertEqual(np.diff(sizes).tolist(), [batch_size] * max_iter)

        # all the batch points are inside the design space
        x_data = np.array([sample.x for sample in state.dataset.samples])
        self.assertTrue(np.all(x_data >= -2.0) and np.all(x_data <= 2.0))


if __name__ == "__main__":
    unittest.main()
