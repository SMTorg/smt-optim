import numpy as np
from scipy._lib.cobyqa.problem import Problem

from .base import BenchmarkProblem


def branin_forrester(x):
    X1 = 15 * x[0] - 5
    X2 = 15 * x[1]

    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    f = (a * (X2 - b * X1 ** 2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x[0]

    return f


class Branin1(BenchmarkProblem):
    """
    [1]
    """

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.bounds = np.array([
            [0, 1],
            [0, 1]
        ])

        self.costs = [0.1, 1]

        self.objective = [self.lf_objective, self.hf_objective]
        self.constraints = [
            [self.lf_constraint, self.hf_constraint]
        ]

        # f_min = 5.5757
        # x_min = np.array([0.9677, 0.2067])

    def hf_objective(self, x):
        return branin_forrester(x)

    def lf_objective(self, x):
        return self.hf_objective(x) - np.cos(0.5*x[0]) - x[1]**3

    def hf_constraint(self, x):
        return -x[0]*x[1] + 0.2

    def lf_constraint(self, x):
        return -x[0]*x[1] - 0.7*x[1] + 0.3*x[0]


# class Branin2:
#     """
#     [1]
#     """
#
#     bounds = np.array([[0, 1], [0, 1]])
#
#     f_min = 12.001
#     f_min_x = np.array([0.941, 0.317])
#
#     def hf_objective(self, x):
#         return branin_forrester(x)
#
#     def lf_objective(self, x):
#         return self.hf_objective(x) + x[0]*x[1] + 0.3*x[0]
#
#     def hf_constraint(self, x):
#
#         x[0] = (x[0] - 0.5)/0.5
#         x[1] = (x[1] - 0.5)/0.5
#
#         return 6 - (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 - x[0]*x[1] - (4*x[1]**2 - 4)*x[1]**2 - 3*np.sin(6 - 6*x[0]) - 3*np.sin(6 - 6*x[1])
#
#     # def hf_constraint(self, x):
#     #
#     #     x1 = (x[0].item() - 1)/2
#     #     x2 = (x[1].item() - 1)/2
#     #
#     #     term1 = 6
#     #     term2 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
#     #     term3 = x1 * x2
#     #     term4 = (4 * x2 ** 2 - 4) * x2 ** 2
#     #     term5 = 3 * np.sin(6 * (1 - x1))
#     #     term6 = 3 * np.sin(6 * (1 - x2))
#     #
#     #     return term1 - term2 - term3 - term4 - term5 - term6
#
#     def lf_constraint(self, x):
#         # Does the perturbation should also be scaled?
#         return self.hf_constraint(x) + x[0] + 0.4*x[1] + np.exp(x[0])
#
# class Sasena1:
#     """
#     [1]
#     """
#
#     bounds = np.array([[0, 5], [0, 5]])
#
#     f_min = -1.1723
#     f_min_x = np.array([2.7450, 2.3523])
#
#     def hf_objective(self, x):
#         return 2+ 0.01*(x[1] - x[0]**2)**2 + (1 - x[0])**2 + 2*(2 - x[1])**2 + 7*np.sin(0.5*x[0])*np.sin(0.7*x[0]*x[1])
#
#     def lf_objective(self, x):
#         return self.hf_objective(x) + np.exp(x[0]) - x[1]**3
#
#     def hf_constraint(self, x):
#         return - np.sin(x[0] - x[1] - np.pi/8)
#
#     def lf_constraint(self, x):
#         return self.hf_constraint(x) + 0.2*x[1] - 0.7*x[0] + x[0]*x[1]
#
# class Sasena2:
#     """
#     [1]
#     """
#
#     bounds = np.array([[0, 1], [0, 1]])
#
#     f_min = -0.7483
#     f_min_x = np.array([0.2017, 0.8332])
#
#     def hf_objective(self, x):
#         return -(x[0] - 1)**2 - (x[1] - 0.5)**2
#
#     def lf_objective(self, x):
#         return self.hf_objective(x) - x[0]*x[1]
#
#     def hf_constraint(self, x):
#         return -(x[0] - 1)**2 - (x[1] - 0.5)**2
#
#     def lf_constraint(self, x):
#         return self.hf_constraint(x) - 0.7*x[0]**3 + x[0]*x[1]**2
#
#     def hf_constraint_1(self, x):
#         return 10*x[0] + x[1] - 7
#
#     def lf_constraint_1(self, x):
#         return self.hf_constraint_1(x) + np.exp(x[0])
#
#     def hf_constraint_2(self, x):
#         return (x[0] - 0.5)**2 + (x[1] - 0.5)**2*np.exp(-x[1]**7) - 12
#
#     def lf_constraint_2(self, x):
#         return self.hf_constraint_2(x) + np.sin(x[1])
#
#
# # TODO: possible add Gomez3 and Toy


class Rosenbrock(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.bounds = np.array([
            [-2, 2],
            [-2, 2]
        ])

        self.costs = [0.1, 1]

        self.objective = [self.lf_objective, self.hf_objective]
        self.constraints = [
            [self.lf_constraint, self.hf_constraint]
        ]

        # f_min = 0.1785
        # f_min_x = np.array([0.5777, 0.3325])


    def hf_objective(self, x):
        res = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        return res

    def lf_objective(self, x):
        res = self.hf_objective(x) + 0.1*np.sin(10*x[0] + 5*x[1])
        return res

    def hf_constraint(self, x):
        res = -x[0]**2 - (x[1] - 1)**1/2
        return -res

    def lf_constraint(self, x):
        res = self.hf_constraint(x) - 0.1*np.sin(10*x[0] + 5*x[1])
        return res


