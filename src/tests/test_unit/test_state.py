import unittest
from functools import partial

import numpy as np

from smt_optim import minimize


class TestSample(unittest.TestCase):

    def test_get_best_sample(self):

        class DummyProb:
            def __init__(self):
                self.f_values = [
                    [14.58319432, 1.90794326, 40.4900085, 112.30641515, 28.04653666, 115.46940515],
                    [1070.4135507, 297.67839692, 1176.605877],
                ]

                self.g_values = [
                    [2.61466648, -0.14862047, -1.15237394, 2.41616628, 0.40703024, 1.38827673],
                    [2.39072941, 0.36316998, 1.48786962],
                ]

                self.objective = [
                    partial(self.f_func, fid=0),
                    partial(self.f_func, fid=1),
                ]

                self.constraints = [
                    [
                        partial(self.g_func, fid=0),
                        partial(self.g_func, fid=1),
                    ]
                ]

            def f_func(self, x, fid):
                return self.f_values[fid].pop(0)

            def g_func(self, x, fid):
                return self.g_values[fid].pop(0)


        problem = DummyProb()

        constraint = [
            {
                "fun": problem.constraints[0],
                "upper": 0.,
            }
        ]

        bounds = np.array([[0, 1]])

        state = minimize(
            problem.objective,
            bounds,
            constraints=constraint,
            costs=[1/10, 1.],
            driver_kwargs={
                "max_iter": 0,
                # "max_budget": 20 * problem.num_dim,
                "nt_init": 3,
                "seed": 0,
            },
            verbose=False
        )

        # no feasible HF sample -> must return HF sample with the lowest constraint violation
        sample = state.get_best_sample()
        self.assertTrue(sample.obj[0] == 297.67839692)
        self.assertTrue(sample.cstr[0] == 0.36316998)

        # feasible LF sample -> must return feasible LF sample with lowest obj value
        sample = state.get_best_sample(fidelity=0)
        self.assertTrue(sample.obj[0] == 1.90794326)
        self.assertTrue(sample.cstr[0] == -0.14862047)


if __name__ == '__main__':
    unittest.main()

