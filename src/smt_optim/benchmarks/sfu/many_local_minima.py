"""
Reference:  https://www.sfu.ca/~ssurjano/optimization.html
"""

import numpy as np
from smt_optim.benchmarks.base import BenchmarkProblem


class Ackley(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Ackley"
        self.num_dim: int | str = "variable"
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 40],
        ])

        self.a = 20
        self.b = 0.2
        self.c = 2*np.pi
        self.exp1 = np.exp(1)

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):

        sum1 = 0
        sum2 = 0

        for i in range(self.num_dim):
            sum1 += x[i]**2
            sum2 += np.cos(self.c*x[i])

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / self.num_dim))
        term2 = -np.exp(sum2 / self.num_dim)

        return term1 + term2 + self.a + self.exp1


class Bukin6(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Bukin6"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-15, 5],
            [-3, 3],
        ])

        self.tags = [
            "sfu",
        ]


    def objective(self, x):

        term1 = 100 * np.sqrt(abs(x[1] - 0.01*x[0]**2))
        term2 = 0.01 * abs(x[0] + 10)

        return term1 + term2


class CrossInTray(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "CrossInTray"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):

        fact1 = np.sin(x[0]) * np.sin(x[1])
        fact2 = np.exp(abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))

        return -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1


class DropWave(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "DropWave"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
            [-10, 10],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        frac1 = 1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))
        frac2 = 0.5 * (x[0] ** 2 + x[1] ** 2) + 2

        return -frac1 / frac2


class EggHolder(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "EggHolder"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-512, 512],
            [-514, 512],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47)))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
        return term1 + term2


class GramacyLee(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "GramacyLee"
        self.num_dim = 1
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [0.5, 2.5],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = np.sin(10 * np.pi * x) / (2 * x)
        term2 = (x - 1) ** 4
        return term1 + term2


class Griewank(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Griewank"
        self.num_dim: int | str = "variable"
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-600, 600],
        ])

        self.tags = [
            "sfu",
            "n_variable",
        ]


    def objective(self, x):
        term1 = 0
        term2 = 1

        for i in range(self.num_dim):
            term1 += x[i] ** 2 / 4000
            term2 *= np.cos(x[i] / np.sqrt(float(i+1)))

        return term1 - term2 + 1


class HolderTable(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "HolderTable"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
            [-10, 10],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        fact1 = np.sin(x[0]) * np.cos(x[1])
        fact2 = np.exp(abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        return -abs(fact1 * fact2)


class Langermann(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Langermann"
        self.num_dim = 2    # could be variable
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 10],
            [0, 10],
        ])

        self.tags = [
            "sfu",
        ]

        self.m = 5
        self.c = np.array([1, 2, 5, 2, 3])
        self.A = np.array([
            [3, 5],
            [5, 2],
            [2, 1],
            [1, 4],
            [7, 9],
        ])

    def objective(self, x):
        outer = 0
        for i in range(self.m):
            inner = 0
            for j in range(self.num_dim):
                inner += (x[j] - self.A[i, j]) ** 2

            new = self.c[i] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
            outer += new
        return outer


class Levy(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Levy"
        self.num_dim: int | str = "variable"
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
        ])

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        w = np.empty(self.num_dim)
        for i in range(self.num_dim):
            w[i] = 1 + (x[i] - 1) / 4

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[self.num_dim-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[self.num_dim-1])) ** 2)

        term2 = 0
        for i in range(self.num_dim-1):
            new = (w[i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[i] + 1)) ** 2)
            term2 += new

        return term1 + term2 + term3


class Levy13(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Levy"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
            [-10, 10],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        term1 = (np.sin(3 * np.pi * x[0])) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1])) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[1])) ** 2)

        return term1 + term2 + term3


class Rastrigin(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Rastrigin"
        self.num_dim: int | str = "variable"
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-5.12, 5.12],
        ])

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        term1 = 0
        for i in range(self.num_dim):
            term1 += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))
        return 10 * self.num_dim + term1


class Schaffer2(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Schaffer2"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-100, 100],
            [-100, 100]
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        fact1 = (np.sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        return 0.5 + fact1 / fact2


class Schaffer4(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Schaffer4"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-100, 100],
            [-100, 100]
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        fact1 = (np.cos(np.sin(abs(x[0] ** 2 - x[1] ** 2)))) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        return 0.5 + fact1 / fact2


class Schwefel(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Schwefel"
        self.num_dim: int | str = "variable"
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-500, 500],
        ])

        self.tags = [
            "sfu",
            "n_variable",
        ]

    def objective(self, x):
        term1 = 0
        for i in range(self.num_dim):
            term1 += x[i] * np.sin(np.sqrt(abs(x[i])))
        return 418.9829 * self.num_dim - term1


class Shubert(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Shubert"
        self.num_dim = 2
        self.num_obj = 1
        self.num_cstr = 0
        self.num_fidelity = 1

        self.bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])

        self.tags = [
            "sfu",
        ]

    def objective(self, x):
        sum1 = 0
        sum2 = 0

        for i in range(5):
            new1 = i * np.cos((i + 1) * x[0] + i)
            new2 = i * np.cos((i + 1) * x[1] + i)
            sum1 += new1
            sum2 += new2

        return sum1 * sum2


