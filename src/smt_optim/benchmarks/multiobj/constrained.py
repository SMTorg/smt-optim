import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class BNH(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name: str = "BNH"

        self.num_dim: int = 2
        self.num_obj: int = 2
        self.num_cstr: int = 2
        self.num_fidelity = 1

        self.tags = [
            "multi-obj",
        ]

        self.bounds = np.array(
            [
                [0, 5],
                [0, 3],
            ]
        )

        # original bounds
        # bounds = np.array([[-15, 30]] * 2)

        self.objective = [
            self.f1,
            self.f2,
        ]

        self.constraints = [
            self.g1,
            self.g2,
        ]

    def f1(self, x):
        return 4 * x[0] ** 2 + 4 * x[1] ** 2

    def f2(self, x):
        return (x[0] - 5) ** 2 + (x[1] - 5) ** 2

    def g1(self, x):
        return (x[0] - 5) ** 2 + x[1] ** 2 - 25

    def g2(self, x):
        return -((x[0] - 8) ** 2) - (x[1] + 3) ** 2 + 7.7


class TNK(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name: str = "TNK"

        self.num_dim: int = 2
        self.num_obj: int = 2
        self.num_cstr: int = 2
        self.num_fidelity = 1

        self.tags = [
            "multi-obj",
        ]

        self.bounds = np.array(
            [
                [0, np.pi],
                [0, np.pi],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]

        self.constraints = [
            self.g1,
            self.g2,
        ]

    def f1(self, x):
        return x[0]

    def f2(self, x):
        return x[1]

    def g1(self, x):
        return -(x[0] ** 2) - x[1] ** 2 + 1 + 0.1 * np.cos(16 * np.arctan(x[0] / x[1]))

    def g2(self, x):
        return (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.5


class OSY(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        self.name: str = "OSY"

        self.num_dim: int = 6
        self.num_obj: int = 2
        self.num_cstr: int = 6
        self.num_fidelity = 1

        self.tags = [
            "multi-obj",
        ]

        self.bounds = np.array(
            [
                [0.0, 10.0],
                [0.0, 10.0],
                [0.0, 5.0],
                [0.0, 6.0],
                [0.0, 5.0],
                [0.0, 10.0],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
        ]

    def f1(self, x):
        return -(
            25 * (x[0] - 2) ** 2
            + (x[1] - 2) ** 2
            + (x[2] - 1) ** 2
            + (x[3] - 4) ** 2
            + (x[4] - 1) ** 2
        )

    def f2(self, x):
        return np.sum(x**2)

    def g1(self, x):
        return -x[0] - x[1] + 2

    def g2(self, x):
        return -6 + x[0] + x[1]

    def g3(self, x):
        return -2 + x[1] - x[0]

    def g4(self, x):
        return -2 + x[0] - 3 * x[1]

    def g5(self, x):
        return -4 + (x[2] - 3) ** 2 + x[3]

    def g6(self, x):
        return -((x[4] - 3) ** 2) - x[5] + 4


if __name__ == "__main__":
    pass
