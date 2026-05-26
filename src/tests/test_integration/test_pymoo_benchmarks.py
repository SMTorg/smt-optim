import unittest

import numpy as np

from smt_optim.benchmarks.base import PymooWrapper
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class TestPymooBenchmarkWrapper(unittest.TestCase):
    def test_no_constraint(self):
        from smt_optim.benchmarks.multiobj.zdt import ZDT1

        problem = ZDT1()
        problem.set_dim(2)

        pymoo_problem = PymooWrapper(problem)
        algorithm = NSGA2(pop_size=100, seed=0)
        res = minimize(pymoo_problem, algorithm, ("n_gen", 100), seed=0)

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.scatter(res.F[:, 0], res.F[:, 1])
        # plt.show()

        self.assertLess(hv(res.F), ref_point[0] * ref_point[1])

    def test_ineq_constraint(self):

        from smt_optim.benchmarks.multiobj.constrained import TNK

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        # solve problem without the constraints
        problem = TNK()

        problem.num_cstr = 0
        problem.constraints = []

        pymoo_problem = PymooWrapper(problem)
        algorithm = NSGA2(pop_size=100, seed=0)
        res_no_cstr = minimize(pymoo_problem, algorithm, ("n_gen", 100), seed=0)
        hv_no_cstr = hv(res_no_cstr.F)

        # solve problem with the constraints
        problem = TNK()

        pymoo_problem = PymooWrapper(problem)
        algorithm = NSGA2(pop_size=100, seed=0)
        res_with_cstr = minimize(pymoo_problem, algorithm, ("n_gen", 100), seed=0)
        hv_with_cstr = hv(res_with_cstr.F)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.scatter(res_no_cstr.F[:, 0], res_no_cstr.F[:, 1])
        # ax.scatter(res_with_cstr.F[:, 0], res_with_cstr.F[:, 1])
        # plt.show()

        # the HV with constraints should be smaller than the one without constraints
        self.assertLess(hv_with_cstr, hv_no_cstr)


if __name__ == "__main__":
    unittest.main()
