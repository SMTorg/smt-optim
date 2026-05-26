import unittest

import numpy as np

from smt_optim.benchmarks.base import PymooWrapper

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


class TestParetoFront(unittest.TestCase):
    def test_constrained_benchmark(self):
        from smt_optim.benchmarks.multiobj.zdt_mf import DTLZ5

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        problem = DTLZ5()
        problem.set_dim(4)

        problem.num_cstr = 0
        problem.constraints = []

        pymoo_prob = PymooWrapper(problem)

        algorithm = NSGA2(pop_size=100, seed=0)
        res_uncstr = minimize(pymoo_prob, algorithm, ("n_gen", 100), seed=0)

        problem = DTLZ5()
        problem.set_dim(4)

        pymoo_prob = PymooWrapper(problem)

        algorithm = NSGA2(pop_size=100, seed=0)
        res_cstr = minimize(pymoo_prob, algorithm, ("n_gen", 100), seed=0)

        self.assertLess(hv.do(res_cstr.F), hv.do(res_uncstr.F))


if __name__ == "__main__":
    unittest.main()
