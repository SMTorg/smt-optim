import numpy as np

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
        self.num_fidelity = 1

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array(
            [
                [0.0, 1.0],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]
        self.set_dim(30)

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def g(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.num_dim - 1.0)

    def h(self, f1: np.ndarray, g: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(f1 / g)

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        f1_val = self.f1(x)
        g_val = self.g(x)
        val = g_val * self.h(f1_val, g_val)
        if ndim == 1:
            return val.item()
        return val


class ZDT2(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1 (with discontinuity)
        # num_dim from reference: 30

        self.name: str = "ZDT2"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0
        self.num_fidelity = 1

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array(
            [
                [0.0, 1.0],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]
        self.set_dim(30)

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def g(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.num_dim - 1.0)

    def h(self, f1: np.ndarray, g: np.ndarray) -> np.ndarray:
        return 1.0 - (f1 / g) ** 2

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        f1_val = self.f1(x)
        g_val = self.g(x)
        val = g_val * self.h(f1_val, g_val)
        if ndim == 1:
            return val.item()
        return val


class ZDT3(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1 (with discontinuity)
        # num_dim from reference: 30

        self.name: str = "ZDT3"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0
        self.num_fidelity = 1

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        self.bounds = np.array(
            [
                [0.0, 1.0],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]
        self.set_dim(30)

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def g(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.num_dim - 1.0)

    def h(self, f1: np.ndarray, g: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        f1_val = self.f1(x)
        g_val = self.g(x)
        val = g_val * self.h(f1_val, g_val)
        if ndim == 1:
            return val.item()
        return val


class ZDT4(BenchmarkProblem):
    def __init__(self):
        super().__init__()

        # pareto-optimal front with g(x) = 1.25 (with discontinuity)
        # num_dim from reference: 10

        self.name: str = "ZDT4"

        self.num_dim: int | str = "variable"
        self.num_obj: int = 2
        self.num_cstr: int = 0
        self.num_fidelity = 1

        self.tags = [
            "n_variable",
            "multi-obj",
            "ZDT",
        ]

        # custom set_dim class method
        self.bounds = np.array(
            [
                [np.nan, np.nan],
            ]
        )

        self.objective = [
            self.f1,
            self.f2,
        ]
        self.set_dim(10)

    def set_dim(self, dim):
        if dim == 1:
            raise Exception("ZDT4 dimension must be greater than 1.")

        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds = np.empty((dim, 2))
            self.bounds[0, :] = [0.0, 1.0]
            self.bounds[1:, :] = [-5.0, 5.0]
        else:
            raise Exception("Not a variable dimension problem.")

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def g(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        return (
            1.0
            + 10.0 * (self.num_dim - 1.0)
            + np.sum(x[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]), axis=1)
        )

    def h(self, f1: np.ndarray, g: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(f1 / g)

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        f1_val = self.f1(x)
        g_val = self.g(x)
        val = g_val * self.h(f1_val, g_val)
        if ndim == 1:
            return val.item()
        return val
