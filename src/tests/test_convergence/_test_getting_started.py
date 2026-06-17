import unittest

import numpy as np

from smt.sampling_methods import LHS
import smt.design_space as ds

from smt_optim import minimize
from smt_optim.benchmarks.registry import get_problem

# Based on the Getting started examples

class TestConvergence(unittest.TestCase):


    # ======= Unconstrained optimization =======
    def test_unconstrained(self):

        def xsinx(x):
            return (x - 3.5) * np.sin((x - 3.5) / (np.pi))

        bounds = np.array([
            [0, 25]
        ])

        state = minimize([xsinx], bounds, max_iter=12, driver_kwargs={"seed": 0}, verbose=False)

        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), -15.12)


    # ======= Constrained optimization =======
    def test_constrained_ineq(self):

        # defines the objective function to minimize
        def modified_branin(x):
            X1 = 15 * x[0] - 5
            X2 = 15 * x[1]
            return (1 * (X2 - (5.1 / (4 * np.pi ** 2)) * X1 ** 2 + (5 / np.pi) * X1 - 6) ** 2 + 10 * (
                        1 - 1 / (8 * np.pi)) * np.cos(X1) + 10) + 5 * x[0]

        # defines the inequality constraint
        def simple_constraint(x):
            return -x[0] * x[1] + 0.2

        # defines the problem bounds
        bounds = np.array([
            [0, 1],
            [0, 1],
        ])

        constraint = [
            {
                "fun": [simple_constraint],
                "upper": 0.,  # equivalent to: g(x) <= 0
            }
        ]

        state = minimize(
            [modified_branin],
            bounds,
            constraints=constraint,
            max_iter=10,
            driver_kwargs={"seed": 0},
            verbose=False,
        )

        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), 5.70)
        self.assertLessEqual(best_sample.cstr.item(), 1e-4)


    def test_constrained_ineq_eq(self):

        # defines the objective function to minimize
        def sasena(x):
            return 2 + 0.01 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 2 * (2 - x[1]) ** 2 + 7 * np.sin(
                0.5 * x[0]) * np.sin(0.7 * x[0] * x[1])

        # defines the equality constraint
        def eq_constraint(x):
            return (x[0] - 2.5) ** 2 + (x[1] - 2.5) ** 2 - 2

        # defines the inequality constraint
        def ineq_constraint(x):
            return -np.sin(x[0] - x[1] - np.pi / 8)

        # defines the problem bounds
        bounds = np.array([
            [0, 5],
            [0, 5]
        ])

        constraints = [
            {
                "fun": [eq_constraint],
                "equal": 0.  # equivalent to: g(x) == 0.
            },
            {
                "fun": [ineq_constraint],
                "lower": -0.8,  # equivalent to: g(x) >= -0.8
                "upper": 0.8,  # equivalent to: g(x) <= 0.8
            }
        ]

        state = minimize(
            [sasena],
            bounds,
            constraints=constraints,
            max_iter=20,
            driver_kwargs={"seed": 0},
            verbose=False,
        )

        best_sample = state.get_best_sample(ctol=1e-6)

        self.assertLessEqual(best_sample.obj.item(), 3.50)
        self.assertLessEqual(best_sample.metadata["rscv"], 1e-6)


    # ======= Multi-fidelity optimization =======
    def test_multi_fidelity(self):

        # high-fidelity function
        def sasena2002_hf(x):
            return -np.sin(x) - np.exp(x / 100) + 10

        # low-fidelity function
        def sasena2002_lf(x):
            return sasena2002_hf(x) + 0.3 + 0.03 * (x - 3) ** 2

        bounds = np.array([
            [0, 10]
        ])

        state = minimize(
            [sasena2002_lf, sasena2002_hf],  # in increasing order of fidelity
            bounds,
            costs=[0.2, 1.],  # in increasing order of fidelity
            max_iter=10,
            driver_kwargs={
                "nt_init": 3,  # number of sample in the initial Design of Experiment (DoE)
                "seed": 0,  # makes this example reproducible
            },
            verbose=False,
        )

        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), 8.10)


    def test_constrained_multi_fidelity(self):

        # defines the high-fidelity objective function
        def rosenbrock_hf(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        # defines the low-fidelity objective function
        def rosenbrock_lf(x):
            return rosenbrock_hf(x) + 0.1 * np.sin(10 * x[0] + 5 * x[1])

        # defines the high-fidelity constraint function
        def constraint_hf(x):
            return -(-x[0] ** 2 - (x[1] - 1) ** 1 / 2)

        # defines the low-fidelity constraint function
        def constraint_lf(x):
            return constraint_hf(x) - 0.1 * np.sin(10 * x[0] + 5 * x[1])

        # defines the problem boundary
        bounds = np.array([
            [-2, 2],
            [-2, 2]
        ])

        # defines the inequality constraint
        constraint = [
            {
                "fun": [constraint_lf, constraint_hf],  # in increasing order of fidelity
                "upper": 0.,  # equivalent to: g(x) <= 0
            }
        ]

        state = minimize(
            [rosenbrock_lf, rosenbrock_hf],  # in increasing order of fidelity
            bounds,
            costs=[0.2, 1.],
            constraints=constraint,
            max_iter=30,
            max_budget=15.,
            # arguments passed to the optimization driver
            driver_kwargs={
                "seed": 0
            },
            # arguments passed to the acquisition strategy
            strategy_kwargs={
                "fidelity_crit": "pessimistic",
                "sp_method": "Cobyla",
                "sp_tol": 1e-4
            },
            verbose=False,
        )

        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), 0.300)


    def test_many_fidelity_levels(self):

        problem = get_problem("Forrester")

        # create custom nested DOE, where each level has the same number of samples
        sampler = LHS(xlimits=problem.bounds, criterion="ese", seed=0)
        doe_hf = sampler(3)
        doe = [doe_hf for _ in range(problem.num_fidelity)]

        state = minimize(
            problem.objective,
            problem.bounds,
            costs=[1 / 1000, 1 / 100, 1 / 10, 1.],
            max_iter=15,
            max_budget=10.,
            # arguments passed to the optimization driver
            driver_kwargs={
                "xt_init": doe,
                "seed": 0,
            },
            strategy_kwargs={
                "var_red_corr": "max",  # more numerically stable with many fidelity levels
            },
            verbose=False,
        )

        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), -5.70)


    # ======= Mixed-variable optimization =======
    def test_mixed_variable(self):

        # fetches the mixed-variable test problem
        problem = get_problem("MixVarBranin")

        # defines the design space using SMT's DesignSpace class
        design_space = ds.DesignSpace([
            ds.FloatVariable(lower=0, upper=1),
            ds.FloatVariable(lower=0, upper=1),
            ds.CategoricalVariable([0, 1]),
            ds.CategoricalVariable([0, 1]),
        ])

        # starts the optimization
        state = minimize(
            [problem.objective],  # list of callables (single callable for mono-fidelity)
            design_space,  # SMT's Design Space object
            max_iter=10,
            driver_kwargs={"seed": 0},
            verbose=False,
        )

        # fetches the best sample
        best_sample = state.get_best_sample()

        self.assertLessEqual(best_sample.obj.item(), -0.888)


if __name__ == "__main__":
    unittest.main()