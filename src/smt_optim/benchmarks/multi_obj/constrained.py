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
                [0.0, 5.0],
                [0.0, 3.0],
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

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = 4.0 * x[:, 0] ** 2 + 4.0 * x[:, 1] ** 2
        if ndim == 1:
            return val.item()
        return val

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = (x[:, 0] - 5.0) ** 2 + (x[:, 1] - 5.0) ** 2
        if ndim == 1:
            return val.item()
        return val

    def g1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = (x[:, 0] - 5.0) ** 2 + x[:, 1] ** 2 - 25.0
        if ndim == 1:
            return val.item()
        return val

    def g2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -((x[:, 0] - 8.0) ** 2) - (x[:, 1] + 3.0) ** 2 + 7.7
        if ndim == 1:
            return val.item()
        return val


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
                [0.0, np.pi],
                [0.0, np.pi],
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

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = x[:, 1]
        if ndim == 1:
            return val.item()
        return val

    def g1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = (
            -(x[:, 0] ** 2)
            - x[:, 1] ** 2
            + 1.0
            + 0.1 * np.cos(16.0 * np.arctan2(x[:, 0], x[:, 1]))
        )
        if ndim == 1:
            return val.item()
        return val

    def g2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 - 0.5
        if ndim == 1:
            return val.item()
        return val


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

    def f1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -(
            25.0 * (x[:, 0] - 2.0) ** 2
            + (x[:, 1] - 2.0) ** 2
            + (x[:, 2] - 1.0) ** 2
            + (x[:, 3] - 4.0) ** 2
            + (x[:, 4] - 1.0) ** 2
        )
        if ndim == 1:
            return val.item()
        return val

    def f2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = np.sum(x**2, axis=1)
        if ndim == 1:
            return val.item()
        return val

    def g1(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -x[:, 0] - x[:, 1] + 2.0
        if ndim == 1:
            return val.item()
        return val

    def g2(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -6.0 + x[:, 0] + x[:, 1]
        if ndim == 1:
            return val.item()
        return val

    def g3(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -2.0 + x[:, 1] - x[:, 0]
        if ndim == 1:
            return val.item()
        return val

    def g4(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -2.0 + x[:, 0] - 3.0 * x[:, 1]
        if ndim == 1:
            return val.item()
        return val

    def g5(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -4.0 + (x[:, 2] - 3.0) ** 2 + x[:, 3]
        if ndim == 1:
            return val.item()
        return val

    def g6(self, x: np.ndarray) -> np.ndarray:
        ndim = x.ndim
        if ndim == 1:
            x = x.reshape(1, -1)
        val = -((x[:, 4] - 3.0) ** 2) - x[:, 5] + 4.0
        if ndim == 1:
            return val.item()
        return val
