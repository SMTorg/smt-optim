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
                [0.0, 1.0],
            ]
        )

        self.objective = [
            [self.f1_lf, self.f1],
            [self.f2_lf, self.f2],
        ]

        self.constraints = [[self.g_lf, self.g]]
        self.set_dim(10)

    def set_dim(self, dim: int):
        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds = np.empty((dim, 2))
            self.bounds[:, 0] = 0.0
            self.bounds[:, 1] = 1.0
        else:
            raise Exception("Not a variable dimension problem.")

    def u(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = np.sum((x - 0.5) ** 2, axis=1)
        if ndim == 1:
            return val.item()
        return val

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        xq = x[:, self.num_obj + 1 :]
        val = (1.0 + self.u(xq)) * np.cos(np.pi / 2.0 * x[:, 0])
        if ndim == 1:
            return val.item()
        return val

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        xq = x[:, self.num_obj + 1 :]
        val = (1.0 + self.u(xq)) * np.sin(np.pi / 2.0 * x[:, 0])
        if ndim == 1:
            return val.item()
        return val

    def g(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = self.f1(x) - 0.5
        if ndim == 1:
            return val.item()
        return val

    def f1_lf(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        xq = x[:, self.num_obj + 1 :]
        val = (1.0 + 0.8 * self.u(xq)) * np.cos(np.pi / 2.0 * x[:, 0])
        if ndim == 1:
            return val.item()
        return val

    def f2_lf(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        xq = x[:, self.num_obj + 1 :]
        val = (1.0 + 1.1 * self.u(xq)) * np.sin(np.pi / 2.0 * x[:, 0])
        if ndim == 1:
            return val.item()
        return val

    def g_lf(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = self.f1_lf(x)
        if ndim == 1:
            return val.item()
        return val
