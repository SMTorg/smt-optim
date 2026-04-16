"""
Reference: Efficient global optimization of constrained mixed variable problems
"""

import numpy as np

import smt.design_space as ds

from smt_optim.benchmarks.base import BenchmarkProblem



# MixVarBranin 4D (2 cont., 2 cat.)
# MixVarAugmentedBranin 12D (10 cont., 2 cat.)
# MixVarAugmentedBranin 12D (10 cont., 2 cat.)
# MixVarGoldstein (2 cont., 2 cat.)

class MixVarBranin(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "MixVarBranin"
        self.num_dim = 4
        self.num_obj = 1
        self.num_cstr = 1

        self.design_space = ds.DesignSpace([
            ds.FloatVariable(0, 1),
            ds.FloatVariable(0, 1),
            ds.CategoricalVariable([0, 1]),
            ds.CategoricalVariable([0, 1]),
        ])

        self.constraints = [self.constraint]


    def h(self, x):
        term1 = 15 * x[1] - (5 / (4 * np.pi ** 2)) * (15 * x[0] - 5) ** 2 \
                + (5 / np.pi) * (15 * x[0] - 5) - 6

        term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x[0] - 5)

        value = (term1 ** 2 + term2 + 10 - 54.8104) / 51.9496
        return value


    def objective(self, x):

        h_val = self.h(x[:2])
        z = x[2:]

        if z[0] == 0 and z[1] == 0:
            return h_val
        elif z[0] == 0 and z[1] == 1:
            return 0.4 * h_val
        elif z[0] == 1 and z[1] == 0:
            return -0.75 * h_val + 3.0
        elif z[0] == 1 and z[1] == 1:
            return -0.5 * h_val + 1.4

        return np.nan

    def constraint(self, x):

        z = x[2:]
        g = np.nan

        if z[0] == 0 and z[1] == 0:
            g = x[0] * x[1] - 0.4
        elif z[0] == 0 and z[1] == 1:
            g = 1.5 * x[0] * x[1] - 0.4
        elif z[0] == 1 and z[1] == 0:
            g = 1.5 * x[0] * x[1] - 0.2
        elif z[0] == 1 and z[1] == 1:
            g = 1.2 * x[0] * x[1] - 0.3

        return -g


class MixVarGoldstein(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "MixVarGoldstein"
        self.num_dim = 4
        self.num_obj = 1
        self.num_cstr = 1

        self.design_space = ds.DesignSpace([
            ds.FloatVariable(0, 100),
            ds.FloatVariable(0, 100),
            ds.CategoricalVariable([0, 1, 2]),
            ds.CategoricalVariable([0, 1, 2]),
        ])

        self.constraints = [self.constraint]

        self.x3_table = np.array([
            [20, 50, 80],
            [20, 50, 80],
            [20, 50, 80],
        ])

        self.x4_table = np.array([
            [20, 20, 20],
            [50, 50, 50],
            [80, 80, 80],
        ])

        self.c1_table = np.array([
            [2, -2, 1],
            [2, -2, 1],
            [2, -2, 1],
        ])

        self.c2_table = np.array([
            [0.5, 0.5, 0.5],
            [-1, -1, -1],
            [-2, -2, -2],
        ])


    def h(self, x):
        x1, x2, x3, x4 = x

        value = (
            53.3108
            + 0.184901 * x1
            - 5.02914e-6 * x1**3
            + 7.72522e-8 * x1**4
            - 0.0870775 * x2
            - 0.106959 * x3
            + 7.98772e-6 * x3**3
            + 0.00242482 * x4
            + 1.32851e-6 * x4**3
            - 0.00146393 * x1 * x2
            - 0.00301588 * x1 * x3
            - 0.00272291 * x1 * x4
            + 0.0017004 * x2 * x3
            + 0.0038428 * x2 * x4
            - 0.000198969 * x3 * x4
            + 1.86025e-5 * x1 * x2 * x3
            - 1.88719e-6 * x1 * x2 * x4
            + 2.50923e-5 * x1 * x3 * x4
            - 5.62199e-5 * x2 * x3 * x4
        )

        return value

    def objective(self, x):
        x1, x2 = x[0], x[1]
        z1, z2 = int(x[2]), int(x[3])

        x3 = self.x3_table[z2, z1]
        x4 = self.x4_table[z2, z1]

        return self.h([x1, x2, x3, x4])

    def constraint(self, x):
        x1, x2 = x[0], x[1]
        z1, z2 = int(x[2]), int(x[3])

        c1 = self.c1_table[z2, z1]
        c2 = self.c2_table[z2, z1]

        value = (
            c1 * np.sin(x1 / 10.0) ** 3
            + c2 * np.cos(x2 / 20.0) ** 2
        )

        return -value


class MultiFidelityMixVarBranin(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "MultiFidelityMixVarBranin"
        self.num_dim = 4
        self.num_obj = 1
        self.num_cstr = 1
        self.num_fidelity = 2

        self.design_space = ds.DesignSpace([
            ds.FloatVariable(0, 1),
            ds.FloatVariable(0, 1),
            ds.CategoricalVariable([0, 1]),
            ds.CategoricalVariable([0, 1]),
        ])


        self.children = MixVarBranin()

        self.objectives = [self.objective_lf, self.children.objective]

        self.constraints = [
            [self.constraint_lf, self.children.constraint]
        ]


    def objective_lf(self, x):
        return self.children.objective(x)  - np.cos(0.5*x[0]) - x[1]**3

    def constraint_lf(self, x):
        return self.children.constraint(x) - 0.1 * np.sin(10*x[0] + 5*x[1])



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Instantiate your problem
    problem = MixVarBranin()

    # Continuous domain of the Branin function
    x1 = np.linspace(0.0, 1.0, 300)
    x2 = np.linspace(0.0, 1.0, 300)

    X1, X2 = np.meshgrid(x1, x2)

    # All 4 combinations of z1, z2
    z_combinations = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    fig = plt.figure(figsize=(14, 10))

    for i, (z1, z2) in enumerate(z_combinations, start=1):

        F = np.zeros_like(X1)

        for r in range(X1.shape[0]):
            for c in range(X1.shape[1]):
                x = np.array([X1[r, c], X2[r, c], z1, z2])
                F[r, c] = problem.objective(x)

        ax = fig.add_subplot(2, 2, i, projection="3d")
        surf = ax.plot_surface(X1, X2, F, cmap="viridis", edgecolor="none")

        ax.set_title(f"z1={z1}, z2={z2}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2, z1, z2)")

        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

    plt.tight_layout()
    plt.show()



