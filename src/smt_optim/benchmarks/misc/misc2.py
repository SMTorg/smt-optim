from functools import partial

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem

from smt_optim.benchmarks.avt311.avt311 import MFRosenbrock

# class MFBarnes(BenchmarkProblem):
#     """
#     Fischer, C. C. (2021). Bayesian Inspired Multi-Fidelity Optimization with Aerodynamic Design.
#     """
#     def __init__(self):
#         super().__init__()
#
#         self.num_dim = 2
#         self.num_cstr = 3
#         self.num_fidelity = 2
#         self.num_obj = 1
#         self.bounds = np.array([
#             [0, 80],
#         ] * self.num_dim)
#
#         self.objective = [
#             # Taylor expansion n=3 around x=(30, 40), -> to add when SMT-optim dependancies include a package with
#             # autodiff functionalities.
#             self.func
#         ]
#
#         self.constraints = [
#             [self.g1_lf, self.g1],
#             [self.g2_lf, self.g2],
#             [self.g3_lf, self.g3],
#         ]
#
#     def func(self, x):
#         return (
#                 75.196
#                 - 3.8112 * x[0]
#                 + 0.12694 * x[0] ** 2
#                 - 2.0567e-3 * x[0] ** 3
#                 + 1.0345e-5 * x[0] ** 4
#                 - 6.8306 * x[1]
#                 + 0.25645 * x[1] ** 2
#                 - 3.4604e-3 * x[1] ** 3
#                 + 1.3514e-5 * x[1] ** 4
#                 + 0.030234 * x[0] * x[1]
#                 - 1.28134e-3 * x[0] * x[1] ** 2
#                 + 3.5256e-5 * x[0] * x[1] ** 3
#                 - 2.266e-7 * x[0] * x[1] ** 4
#                 + 3.4054e-4 * x[0] ** 2 * x[1]
#                 - 5.2375e-6 * x[0] ** 2 * x[1] ** 2
#                 - 6.3e-8 * x[0] ** 2 * x[1] ** 3
#                 - 1.6638e-6 * x[0] ** 3 * x[1]
#                 + 7.0e-10 * x[0] ** 3 * x[1] ** 3
#                 - 28.106 / (x[1] + 1.0)
#                 - 2.8673 * np.exp(5.0e-4 * x[0] * x[1])
#         )
#
#     def g1(self, x):
#         return 1 - x[0]*x[1]/700
#
#     def g2(self, x):
#         return x[0]**2/625 - x[1]/5
#
#     def g3(self, x):
#         return (x[0]/500 - 0.11) - (x[1]/50 - 1)**2
#
#     def g1_lf(self, x):
#         return (50 - x[0] - x[1])/10
#
#     def g2_lf(self, x):
#         return (0.64*x[0] - x[1])/6
#
#     def g3_lf(self, x):
#         if x[1] > 50:
#             return 0.34 + 0.006*x[0] - 0.0134*x[1]
#         else:
#             return 0.0134*x[1] + 0.006*x[0] - 1


class MFConstraintRosenbrock(BenchmarkProblem):
    """
    Fischer, C. C. (2021). Bayesian Inspired Multi-Fidelity Optimization with Aerodynamic Design.
    """

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array(
            [
                [-2, 2],
            ]
            * self.num_dim
        )

        self.ref_problem = MFRosenbrock()

        self.objective = [
            partial(self.ref_problem.f, fid=0),
            partial(self.ref_problem.f, fid=2),
        ]

        self.constraints = [[self.g1_lf, self.g1]]

        self.tags = ["n_variable"]

    def set_dim(self, dim):
        self.ref_problem.set_dim(dim)
        self.num_dim = dim
        self.bounds = np.array([[-2, 2]] * dim)

    def g1(self, x):
        return x[0] ** 2 + np.sum(x[1:] - 1) / 2

    def g1_lf(self, x):
        return self.g1(x) + 0.1 * np.sin(10 * x[0] + np.sum(5 * x[1:]))


if __name__ == "__main__":
    from smt_optim.subsolvers.multistart import multistart_minimize

    prob = MFConstraintRosenbrock()
    prob.set_dim(2)

    constraints = [
        {
            "fun": lambda x, f=prob.g1: -f(x),
            "type": "ineq",
        },
        # {
        #     "fun": lambda x, f=prob.g2: -f(x),
        #     "type": "ineq",
        # },
        # {
        #     "fun": lambda x, f=prob.g3: -f(x),
        #     "type": "ineq",
        # },
    ]

    # res = so.minimize(prob.objective[-1], x0=np.array([0., 0.]), bounds=prob.bounds, constraints=constraints)
    res = multistart_minimize(
        prob.objective[-1], bounds=prob.bounds, constraints=constraints
    )

    print(res.x)
    print(res.fun)
    print(prob.g1(res.x))
    # print(res)

    # import matplotlib.pyplot as plt
    # from smt_optim.utils.plot_2d import get_plot2d_data
    #
    # prob.set_dim(2)
    # XX, YY, ZZ = get_plot2d_data(prob.constraints[0][0], prob.bounds)
    #
    # fig, ax = plt.subplots()
    # ax.contourf(XX, YY, np.where(ZZ<=0, np.nan, ZZ))
    # plt.show()
