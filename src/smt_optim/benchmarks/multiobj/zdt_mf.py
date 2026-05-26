"""
Reference: Towards a multi-fidelity & multi-objective Bayesian optimization efficient algorithm
Rémy Charayron, Thierry Lefebvre, Nathalie Bartoli, Joseph Morlier

With multi-fidelity variant?
ZDT1, ZDT2, ZDT3, ZDT5 (w/ cstr)

DTLZ5

"""

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class DTLZ5(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name: str = "DTLZ5"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 1
        self.num_fidelity: int = 2

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array(
            [
                [0, 1],
            ]
        )

        self.objective = [
            [self.f1_lf, self.f1],
            [self.f2_lf, self.f2],
        ]

        self.constraints = [[self.g_lf, self.g]]

    def u(self, x: np.ndarray):
        return np.sum((x - 0.5) ** 2)

    def f1(self, x):
        xq = x[self.num_obj + 1 :]
        return (1 + self.u(xq)) * np.cos(np.pi / 2 * x[0])

    def f2(self, x):
        xq = x[self.num_obj + 1 :]
        return (1 + self.u(xq)) * np.sin(np.pi / 2 * x[0])

    def g(self, x):
        return self.f1(x) - 0.5

    def f1_lf(self, x):
        xq = x[self.num_obj + 1 :]
        return (1 + 0.8 * self.u(xq)) * np.cos(np.pi / 2 * x[0])

    def f2_lf(self, x):
        xq = x[self.num_obj + 1 :]
        return (1 + 1.1 * self.u(xq)) * np.sin(np.pi / 2 * x[0])

    def g_lf(self, x):
        return self.f1_lf(x)
