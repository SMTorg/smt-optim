import unittest

import numpy as np

from smtoptim.subsolvers.multistart import multistart_minimize, MultistartResult



class TestMultistartMinimize(unittest.TestCase):

    """
    To be tested:
    - returns valid structure
    - best solution correctly selected
    - reproducibility
    - uses provided initial points
    - bounds are respected
    - single start edge case
    - constraint handling

    """


    def test_returns_valid_structure(self):

        def f(x):
            return np.sum(x**2)

        bounds = np.array([[-1, 1], [-1, 1]])

        res = multistart_minimize(
            f,
            bounds,
            n_start=5,
            seed=42
        )

        self.assertIsInstance(res, MultistartResult)

        self.assertEqual(res.multi_x0.shape, (5, 2))
        self.assertEqual(res.multi_x.shape, (5, 2))
        self.assertEqual(res.multi_f.shape, (5,))
        self.assertEqual(len(res.multi_sp_res), 5)


    def test_selects_best_solution(self):

        def f(x):
            return (x[0] - 0.2)**2

        bounds = np.array([[0.0, 1.0]])

        res = multistart_minimize(
            f,
            bounds,
            n_start=10,
            seed=1
        )

        self.assertAlmostEqual(res.x[0], 0.2, places=3)
        self.assertAlmostEqual(res.fun, f(res.x), places=10)


    def test_reproducibility(self):

        def rastrigin(x: np.ndarray) -> np.ndarray:

            ndim = x.ndim

            if ndim == 1:
                x = x.reshape(1, -1)

            n = x.shape[1]
            A = 10

            def temp(x):
                return x ** 2 - A * np.cos(2 * np.pi * x)

            temp_vec = np.vectorize(temp)
            value = A * n + np.sum(temp_vec(x), axis=1)

            if ndim == 1:
                value = value.item()

            return value

        bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])

        res1 = multistart_minimize(rastrigin, bounds, seed=42)
        res2 = multistart_minimize(rastrigin, bounds, seed=42)

        np.testing.assert_allclose(res1.multi_x0, res2.multi_x0)
        np.testing.assert_allclose(res1.multi_x, res2.multi_x)


    def test_uses_provided_initial_points(self):

        def f(x):
            return np.sum(x ** 2)

        bounds = np.array([[-1, 1]])

        multi_x0 = np.array([[0.5], [-0.5]])

        res = multistart_minimize(
            f,
            bounds,
            multi_x0=multi_x0
        )

        np.testing.assert_allclose(res.multi_x0, multi_x0)


    def test_solution_within_bounds(self):

        def f(x):
            return x[0] ** 2

        bounds = np.array([[0.1, 1.0]])

        res = multistart_minimize(
            f,
            bounds,
            n_start=5
        )

        self.assertTrue(np.all(res.multi_x >= 0.0))
        self.assertTrue(np.all(res.multi_x <= 1.0))


    def test_single_start(self):

        def f(x):
            return x[0] ** 2

        bounds = np.array([[-1, 1]])

        res = multistart_minimize(f, bounds, n_start=1)

        self.assertEqual(res.multi_x.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
