import unittest
import numpy as np

from smt_optim.acquisition_strategies.mfsego import build_scipy_constraints
from dataclasses import dataclass


@dataclass
class DummyConfig:
    equal: float | None = None
    lower: float | None = None
    upper: float | None = None


class DummyProblem:
    def __init__(self, configs):
        self.cstr_configs = configs


class DummyModel:
    def __init__(self, mu, var):
        self.mu = np.array([[mu]])
        self.var = np.array([[var]])

    def predict_values(self, x):
        return self.mu

    def predict_variances(self, x):
        return self.var


class DummyState:
    def __init__(self, configs, equal_vals, lower_vals, upper_vals, models):
        self.problem = DummyProblem(configs)
        self.cstr_equal = equal_vals
        self.cstr_lower = lower_vals
        self.cstr_upper = upper_vals
        self.cstr_models = models


class TestConstraintsRelaxation(unittest.TestCase):
    def test_scipy_constraints_relaxation(self):
        # We test a point with mu=2.0 and var=4.0 (so s=2.0)
        m = DummyModel(2.0, 4.0)
        x = np.array([[0.0]])

        # Upper bound: g(x) <= 1.0
        # Relaxed should be: g(x) - 3s <= 1.0  => 2.0 - 3*2.0 = -4.0 <= 1.0 (True!)
        # Scipy ineq: 1.0 - (-4.0) = 5.0 >= 0
        state_upper = DummyState([DummyConfig(upper=1.0)], [None], [None], [1.0], [m])
        cstrs = build_scipy_constraints(state_upper, relax=3.0)
        self.assertEqual(cstrs[0]["type"], "ineq")
        val = cstrs[0]["fun"](x)
        self.assertAlmostEqual(val, 5.0)

        # Lower bound: g(x) >= 10.0
        # Relaxed should be: g(x) + 3s >= 10.0 => 2.0 + 3*2.0 = 8.0 >= 10.0 (False, violation)
        # Scipy ineq: 8.0 - 10.0 = -2.0 >= 0
        state_lower = DummyState([DummyConfig(lower=10.0)], [None], [10.0], [None], [m])
        cstrs = build_scipy_constraints(state_lower, relax=3.0)
        self.assertEqual(cstrs[0]["type"], "ineq")
        val = cstrs[0]["fun"](x)
        self.assertAlmostEqual(val, -2.0)

        # Equal bound: g(x) == 1.0
        # Relaxed should be: |g(x) - 1.0| <= 3s => |2.0 - 1.0| = 1.0 <= 6.0 (True!)
        # Scipy ineq: -1.0 + 6.0 = 5.0 >= 0
        state_eq = DummyState([DummyConfig(equal=1.0)], [1.0], [None], [None], [m])
        cstrs = build_scipy_constraints(state_eq, relax=3.0)
        self.assertEqual(cstrs[0]["type"], "ineq")
        val = cstrs[0]["fun"](x)
        self.assertAlmostEqual(val, 5.0)

    def test_scipy_constraints_no_relaxation(self):
        m = DummyModel(2.0, 4.0)
        x = np.array([[0.0]])

        # Upper bound: g(x) <= 1.0
        # Normal: 2.0 <= 1.0 (False) => Scipy ineq: 1.0 - 2.0 = -1.0 >= 0
        state_upper = DummyState([DummyConfig(upper=1.0)], [None], [None], [1.0], [m])
        cstrs = build_scipy_constraints(state_upper, relax=0.0)
        self.assertEqual(cstrs[0]["type"], "ineq")
        self.assertAlmostEqual(cstrs[0]["fun"](x), -1.0)

        # Equal bound: g(x) == 1.0
        # Normal: 2.0 == 1.0 (False) => Scipy eq: 2.0 - 1.0 = 1.0 == 0
        state_eq = DummyState([DummyConfig(equal=1.0)], [1.0], [None], [None], [m])
        cstrs = build_scipy_constraints(state_eq, relax=0.0)
        self.assertEqual(cstrs[0]["type"], "eq")
        self.assertAlmostEqual(cstrs[0]["fun"](x), 1.0)


if __name__ == "__main__":
    unittest.main()
