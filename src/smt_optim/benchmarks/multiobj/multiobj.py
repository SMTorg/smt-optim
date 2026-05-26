"""
Reference: Towards a multi-fidelity & multi-objective Bayesian optimization efficient algorithm
Rémy Charayron, Thierry Lefebvre, Nathalie Bartoli, Joseph Morlier

With multi-fidelity variant?
ZDT1, ZDT2, ZDT3, ZDT5 (w/ cstr)

DTLZ5

"""

import numpy as np

import smt.design_space as ds

from smt_optim.benchmarks.base import BenchmarkProblem


class ZDT1(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1 (with discontinuity)
        # num_dim from reference: 30

        self.name: str = "ZDT1"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array([
            [0, 1],
        ])

        self.objectives = [
            self.f1, self.f2,
        ]

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + 9 * np.sum(x[1:])/(self.num_dim-1)

    def h(self, f1, g):
        return 1 - np.sqrt(f1/g)

    def f2(self, x):
        f1 = self.f1(x)
        g = self.g(x)
        return g*self.h(f1, g)


class ZDT2(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1 (with discontinuity)
        # num_dim from reference: 30

        self.name: str = "ZDT2"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array([
            [0, 1],
        ])

        self.objectives = [
            self.f1, self.f2,
        ]

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + 9 * np.sum(x[1:])/(self.num_dim-1)

    def h(self, f1, g):
        return 1 - (f1/g)**2

    def f2(self, x):
        f1 = self.f1(x)
        g = self.g(x)
        return g*self.h(f1, g)


class ZDT3(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1 (with discontinuity)
        # num_dim from reference: 30

        self.name: str = "ZDT3"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array([
            [0, 1],
        ])

        self.objectives = [
            self.f1, self.f2,
        ]

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + 9 * np.sum(x[1:])/(self.num_dim-1)

    def h(self, f1, g):
        return 1 - np.sqrt(f1/g) - f1/g*np.sin(10*np.pi*f1)

    def f2(self, x):
        f1 = self.f1(x)
        g = self.g(x)
        return g*self.h(f1, g)


class ZDT4(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1.25 (with discontinuity)
        # num_dim from reference: 10

        self.name: str = "ZDT4"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        # custom set_dim class method
        self.bounds = np.array([
            [np.nan, np.nan],
        ])

        self.objectives = [
            self.f1, self.f2,
        ]

    def set_dim(self, dim):

        if dim == 1:
            raise Exception("ZDT4 dimension must be greater than 1.")

        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds = np.empty((dim, 2))
            self.bounds[0, :] = [0, 1]
            self.bounds[1:, :] = [-5, 5]
        else:
            raise Exception("Not a variable dimension problem.")

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + 10*(self.num_dim-1) + np.sum(x[1:] - 10*np.cos(4*np.pi*x[1:]))

    def h(self, f1, g):
        return 1 - np.sqrt(f1/g)

    def f2(self, x):
        f1 = self.f1(x)
        g = self.g(x)
        return g*self.h(f1, g)


    # TODO: implement ZDT45 and ZDT46


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from smt_optim.utils.plot_2d import get_plot2d_data

    # Instantiate your problem
    problem = ZDT4()

    problem.set_dim(2)

    x1 = np.linspace(0, 1, 101)
    x2 = np.linspace(0, 0, 101)

    XX, YY = np.meshgrid(x1, x2)

    data = np.vstack((XX.ravel(), YY.ravel())).T
    f1 = np.empty(data.shape[0])
    f2 = np.empty(data.shape[0])

    for i in range(data.shape[0]):
        f1[i] = problem.f1(data[i, :])
        f2[i] = problem.f2(data[i, :])

    F1 = f1.reshape(XX.shape)
    F2 = f2.reshape(XX.shape)

    fig, ax = plt.subplots()
    ax.scatter(F1, F2, 5)

    plt.show()



