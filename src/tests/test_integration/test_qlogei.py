import unittest

import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, DriverConfig, Driver

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import qLogEI


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


class TestqLogEI(unittest.TestCase):
    def run_driver(self, batch_size, max_iter):
        obj_config = ObjectiveConfig(objective=[branin], surrogate=SmtAutoModel)

        problem = Problem(
            obj_configs=[obj_config],
            design_space=np.array([[-5.0, 10.0], [0.0, 15.0]]),
        )

        driver = Driver(
            problem=problem,
            config=DriverConfig(max_iter=max_iter, nt_init=6, seed=42),
            strategy=qLogEI,
            strategy_kwargs=dict(
                batch_size=batch_size,
                n_samples=64,
                n_start=2,
            ),
        )

        return driver.optimize()

    def test_qlogei_batch_size(self):
        batch_size = 3
        max_iter = 2

        state = self.run_driver(batch_size, max_iter)

        # each iteration adds batch_size samples
        self.assertEqual(len(state.dataset.samples), 6 + batch_size * max_iter)

    def test_qlogei_seed(self):
        state_1 = self.run_driver(batch_size=2, max_iter=2)
        state_2 = self.run_driver(batch_size=2, max_iter=2)

        x_1 = np.array([sample.x for sample in state_1.dataset.samples])
        x_2 = np.array([sample.x for sample in state_2.dataset.samples])

        np.testing.assert_array_equal(x_1, x_2)

    def test_qlogei_branin(self):
        state = self.run_driver(batch_size=3, max_iter=10)

        f_min = min(sample.obj[0] for sample in state.dataset.samples)

        # global minimum of Branin: 0.398
        self.assertLessEqual(f_min, 1.0)


if __name__ == "__main__":
    unittest.main()
