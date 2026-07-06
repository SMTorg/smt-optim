"""
Reference:  https://www.sfu.ca/~ssurjano/optimization.html
"""

import numpy as np
from smt_optim.benchmarks.base import BenchmarkProblem


class Bohachevsky1(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Bohachevsky1"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-100, 100],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = x[0] ** 2
        term2 = 2 * x[1] ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x[0])
        term4 = -0.4 * np.cos(4 * np.pi * x[1])
        return term1 + term2 + term3 + term4 + 0.7


class Bohachevsky2(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Bohachevsky2"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-100, 100],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = x[0] ** 2
        term2 = 2 * x[1] ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])
        return term1 + term2 + term3 + 0.3


class Bohachevsky3(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Bohachevsky3"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-100, 100],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = x[0] ** 2
        term2 = 2 * x[1] ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x[0] + 4 * np.pi * x[1])
        return term1 + term2 + term3 + 0.3


class Perm(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Perm"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-self.num_dim, self.num_dim],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

        self.b = 10

    def set_dim(self, dim: int):
        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds[:, 0] = -self.num_dim
            self.bounds[:, 1] = self.num_dim

    def objective(self, x):

        outer = 0
        for i in range(self.num_dim):
            inner = 0
            for j in range(self.num_dim):
                inner += ((j + 1) + self.b) * (
                    x[j] ** (i + 1) - (1 / (j + 1)) ** (i + 1)
                )

            outer += inner**2

        return outer


class RotatedHyperEllipsoid(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "RotatedHyperEllipsoid"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-65.536, 65.536],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):

        outer = 0.0
        for i in range(self.num_dim):
            outer += np.sum(x[: i + 1] ** 2)

        return outer


class Sphere(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Sphere"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-5.12, 5.12],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        return np.sum(x**2)


class SumDifferentPowers(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "SumDifferentPowers"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-1, 1],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        powers = np.linspace(1, self.num_dim, self.num_dim) + 1
        return np.sum(np.abs(x) ** powers)


class SumSquares(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "SumSquares"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-10, 10],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        indices = np.linspace(1, self.num_dim, self.num_dim)
        return np.sum(indices + x**2)


class Trid(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name = "Trid"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array(
            [
                [-(self.num_dim**2), self.num_dim**2],
            ]
            * self.num_dim
        )

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def set_dim(self, dim: int):
        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds[:, 0] = -(self.num_dim**2)
            self.bounds[:, 1] = self.num_dim**2

    def objective(self, x):
        term1 = np.sum((x - 1) ** 2)
        term2 = np.sum(x[1:] * x[:-1])
        return term1 - term2
