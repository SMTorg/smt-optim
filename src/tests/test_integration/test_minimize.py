import unittest
from smt_optim.benchmarks.registry import get_problem
from smt_optim.frameworks import minimize


MAX_ITER = 2
N_START = 1

class TestMinimizeAPI(unittest.TestCase):

    def test_ego(self):

        problem = get_problem("Branin1")

        state = minimize(
            [problem.objective[0]],
            design_space=problem.bounds,
            method="ego",
            max_iter=MAX_ITER,
            verbose=False,
            strategy_kwargs={"n_start": N_START, }
        )


    def test_sego(self):

        problem = get_problem("Sasena1")

        def cstr2(x):
            return (x[0] - 2.5) ** 2 + (x[1] - 2.5) ** 2 - 2

        constraint = [
            {
                "fun": [problem.constraints[0][-1]],
                "lower": -0.8,
                "upper": 0.8,
            },
            {
                "fun": [cstr2],
                "equal": 0.0,
            }
        ]

        state = minimize(
            [problem.objective[0]],
            design_space=problem.bounds,
            method="sego",
            constraints=constraint,
            max_iter=MAX_ITER,
            verbose=False,
            strategy_kwargs={"n_start": N_START, }
        )



    def test_mfsego(self):

        problem = get_problem("Branin1")

        constraint = [
            {
                "fun": problem.constraints[0],
                "upper": 0.,
            }
        ]

        state = minimize(
            problem.objective,
            design_space=problem.bounds,
            method="mfsego",
            costs=[0.2, 1.],
            constraints=constraint,
            max_iter=MAX_ITER,
            verbose=False,
            strategy_kwargs={"n_start": N_START, }
        )


    def test_vfpi(self):


        problem = get_problem("Branin1")

        constraint = [
            {
                "fun": problem.constraints[0],
                "upper": 0.,
            }
        ]

        state = minimize(
            problem.objective,
            design_space=problem.bounds,
            method="vfpi",
            costs=[0.2, 1.],
            constraints=constraint,
            max_iter=MAX_ITER,
            verbose=False,
            strategy_kwargs={"n_start": N_START,}
        )


if __name__ == '__main__':
    unittest.main()