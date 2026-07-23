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

###############################################################################
# ZDT1 Benchmark
###############################################################################

class ZDT1(BenchmarkProblem):
    def __init__(self):
        super().__init__()
        self.name = "ZDT1"
        self.num_dim = 30
        self.num_obj = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.tags = ["n_variable", "multi-obj", "multi-fidelity"]
        
        # Corrected bounds for 30 dimensions
        self.bounds = np.array(
            [
                [0, 1],
            ]
        )
        
        self.objective = [
            [self.f1_lf, self.f1],
            [self.f2_lf, self.f2],
        ]
        self.constraints = None

    def g(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def g_lf(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def f1(self, x):
        return x[0]

    def f1_lf(self, x):
        return x[0] * 0.9 + 0.1

    def h(self, f1, g):
        return 1 - np.sqrt(f1 / g)

    def f2(self, x):
        g_val = self.g(x)
        f1_val = self.f1(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def f2_lf(self, x):
        g_val = self.g_lf(x)
        f1_val = self.f1_lf(x)
        h_val = self.h(f1_val, g_val)
        return (0.8 * g_val - 0.2) * (1.2 * h_val + 0.2)


###############################################################################
# ZDT2 Benchmark
###############################################################################

class ZDT2(BenchmarkProblem):
    def __init__(self):
        super().__init__()
        self.name = "ZDT2"
        self.num_dim = 30
        self.num_obj = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.tags = ["n_variable", "multi-obj", "multi-fidelity"]
        
        self.bounds = np.array(
            [
                [0, 1],
            ]
        )
        
        self.objective = [
            [self.f1_lf, self.f1],
            [self.f2_lf, self.f2],
        ]
        self.constraints = None

    def g(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def g_lf(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def f1(self, x):
        return x[0]

    def f1_lf(self, x):
        return 0.8 * x[0] + 0.2

    def h(self, f1, g):
        return 1 - (f1 / g) ** 2

    def f2(self, x):
        g_val = self.g(x)
        f1_val = self.f1(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def f2_lf(self, x):
        g_val = self.g_lf(x)
        f1_val = self.f1_lf(x)
        h_val = self.h(f1_val, g_val)
        # Formula derived from ZDT2_LF functional snippet
        return (0.9 * g_val + 0.2) * (1.1 * h_val - 0.2)


###############################################################################
# ZDT3 Benchmark
###############################################################################

class ZDT3(BenchmarkProblem):
    def __init__(self):
        super().__init__()
        self.name = "ZDT3"
        self.num_dim = 30
        self.num_obj = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.tags = ["n_variable", "multi-obj", "multi-fidelity"]
        
        self.bounds = np.array(
            [
                [0, 1],
            ]
        )
        
        self.objective = [
            [self.f1_lf, self.f1],
            [self.f2_lf, self.f2],
        ]
        self.constraints = None

    def g(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def g_lf(self, x):
        return 1 + 9 * np.sum(x[1:]) / (self.num_dim - 1)

    def f1(self, x):
        return x[0]

    def f1_lf(self, x):
        return 0.75 * x[0] + 0.25

    def h(self, f1, g):
        # ZDT3 specific h function
        return 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)

    def f2(self, x):
        g_val = self.g(x)
        f1_val = self.f1(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def f2_lf(self, x):
        g_val = self.g_lf(x)
        f1_val = self.f1_lf(x)
        h_val = self.h(f1_val, g_val)
        # Formula derived from ZDT3_LF functional snippet
        return g_val * (1.25 * h_val - 0.25)