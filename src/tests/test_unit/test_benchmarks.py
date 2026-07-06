import unittest

import numpy as np
import scipy.stats as stats

import smt.design_space as ds
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

from smt_optim.benchmarks.registry import list_problems, get_problem


class TestBenchmarkProblems(unittest.TestCase):
    def test_all_benchmark_attributes(self):

        required_attributes = [
            "name",
            "num_dim",
            "num_obj",
            "num_cstr",
            "num_fidelity",
            # "bounds",
            # "objective",
            # "constraints",
        ]

        problems = list_problems(
            num_dim=None,
            num_obj=None,
            num_cstr=None,
            num_fidelity=None,
        )

        for prob in problems:
            with self.subTest(problem=prob.__class__.__name__):
                # test that prob has all the required attributes -> .num_dim, num_cstr, num_obj, etc.
                for attr in required_attributes:
                    value = getattr(prob, attr)

                    self.assertIsNotNone(
                        value,
                        msg=f"{prob.__class__.__name__}: '{attr}' was not implemented",
                    )

                # test that it has the corresponding number of objective, constraints and fidelities.

                # test num_dim
                if isinstance(prob.num_dim, int):
                    pass
                elif prob.num_dim == "variable":
                    prob.set_dim(2)
                    if not isinstance(prob.num_obj, int):
                        raise NotImplementedError()
                else:
                    raise TypeError()

                # test bounds
                if prob.bounds is not None:
                    self.assertEqual(
                        prob.bounds.shape,
                        (prob.num_dim, 2),
                        msg=f"{prob.name}: bounds should have shape "
                        f"({prob.num_dim}, 2)",
                    )

                    lower = prob.bounds[:, 0]
                    upper = prob.bounds[:, 1]

                    self.assertTrue(
                        np.all(lower < upper),
                        msg=f"{prob.name}: lower bounds must be < upper bounds",
                    )

                else:
                    self.assertIsInstance(prob.design_space, ds.DesignSpace)

                # test num_obj
                # single objective
                if prob.num_obj == 1:
                    if prob.num_fidelity == 1:
                        self.assertTrue(callable(prob.objective))
                    else:
                        self.assertEqual(len(prob.objective), prob.num_fidelity)

                # multi-objective
                elif prob.num_obj > 1:
                    self.assertIsInstance(prob.objective, list)
                    self.assertEqual(len(prob.objective), prob.num_obj)

                # num_obj < 1 ?
                else:
                    raise ValueError()

                # test_num_cstr
                if prob.num_cstr > 0:
                    self.assertEqual(len(prob.constraints), prob.num_cstr)
                else:
                    self.assertTrue(prob.constraints is None)

                if prob.num_obj == 1:
                    objective = [prob.objective]
                else:
                    objective = prob.objective

                # test_fidelity
                if prob.num_fidelity > 1:
                    for obj_idx in range(prob.num_obj):
                        self.assertEqual(len(objective[obj_idx]), prob.num_fidelity)

                    for cstr_idx in range(prob.num_cstr):
                        self.assertEqual(
                            len(prob.constraints[cstr_idx]), prob.num_fidelity
                        )

                # test that all objectives and constraints are callables and respect their domain.
                num_sample = 3

                if isinstance(prob.bounds, np.ndarray):
                    sampler = stats.qmc.LatinHypercube(d=prob.num_dim, seed=0)
                    doe = sampler.random(num_sample)
                    doe = stats.qmc.scale(doe, prob.bounds[:, 0], prob.bounds[:, 1])

                elif isinstance(prob.design_space, ds.DesignSpace):
                    # sampler = LHS(xlimits=prob.design_space.get_unfolded_num_bounds(),
                    #               criterion="ese",
                    #               seed=0)
                    # doe = sampler(num_sample)

                    sampler = MixedIntegerSamplingMethod(
                        LHS, prob.design_space, criterion="ese", seed=0
                    )
                    doe = sampler(10)

                else:
                    raise TypeError()

                for idx in range(doe.shape[0]):
                    for lvl in range(prob.num_fidelity):
                        for obj_idx in range(prob.num_obj):
                            if prob.num_fidelity == 1:
                                val = objective[obj_idx](doe[idx, :])
                            else:
                                val = objective[obj_idx][lvl](doe[idx, :])

                            is_scalar = isinstance(val, float)
                            is_np_1d = isinstance(val, np.ndarray) and val.ndim == 1
                            self.assertTrue(is_scalar or is_np_1d)

                        for cstr_idx in range(prob.num_cstr):
                            if prob.num_fidelity == 1:
                                val = prob.constraints[cstr_idx](doe[idx, :])
                            else:
                                val = prob.constraints[cstr_idx][lvl](doe[idx, :])

                            is_scalar = isinstance(val, float)
                            is_np_1d = isinstance(val, np.ndarray) and val.ndim == 1
                            self.assertTrue(is_scalar or is_np_1d)

    def test_get_problem_returns_independent_instances(self):
        prob1 = get_problem("Alos")
        prob2 = get_problem("Alos")

        # Different objects
        self.assertIsNot(prob1, prob2)

        # Mutate one instance
        original = prob2.num_dim
        prob1.num_dim = 999

        # Other instance is unchanged
        self.assertIsNot(prob1, prob2)
        self.assertEqual(prob2.num_dim, original)

        prob1.tags.append("new_tag")
        self.assertNotIn("new_tag", prob2.tags)

    def test_list_problems_returns_independent_instances(self):
        probs1 = list_problems()
        probs2 = list_problems()

        # Lists themselves are different objects
        self.assertIsNot(probs1, probs2)

        # Problem instances are different objects
        for p1, p2 in zip(probs1, probs2):
            self.assertIsNot(p1, p2)


if __name__ == "__main__":
    unittest.main()
