import numpy as np
from smt_optim.benchmarks.base import BenchmarkProblem

class Gano(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 1
        self.num_fidelity = 2
        self.bounds = np.array([
            [0.1, 10],
            [0.1, 10]
        ])

        self.costs = [0.25, 1]

        self.objective = [self.gano_2b_f, self.gano_2a_f]
        self.constraints = [
            [self.gano_2b_g, self.gano_2a_g]
        ]

    def gano_2a_f(self, x: np.ndarray) -> np.ndarray:
        return 4 * x[0] ** 2 + x[1] ** 3 + x[0] * x[1]

    def gano_2a_g(self, x: np.ndarray) -> np.ndarray:
        return 1 / x[0] + 1 / x[1] - 2

    def gano_2b_f(self, x: np.ndarray) -> np.ndarray:
        return 4 * (x[0] + 0.1) ** 2 + (x[1] + 0.1) ** 3 + x[0] * x[1] + 0.1

    def gano_2b_g(self, x: np.ndarray) -> np.ndarray:
        return 1 / x[0] + 1 / (x[1] + 0.1) - 2 - 0.001


class G8(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 2
        self.num_fidelity = 2
        self.bounds = np.array([
            [0, 10],
            [0, 10]
        ])

        self.objective = [self.G8b_f, self.G8a_f]
        self.constraints = [
            [self.G8b_g1, self.G8a_g1],
            [self.G8b_g2, self.G8a_g2],
        ]

    def G8a_f(self, x: np.ndarray) -> np.ndarray:

        if x[0] == 0:
            return 0

        return -np.sin(2*np.pi*x[0])**3*np.sin(2*np.pi*x[1])/(x[0]**3*(x[0]+x[1]))

    def G8b_f(self, x: np.ndarray) -> np.ndarray:
        x[0] += 0.1
        x[1] -= 0.1
        return G8a_f(x)

        # if x[0]+0.1 == 0:
        #     return 0
        #
        # return -np.sin(2*np.pi*(x[0] + 0.1))**3*np.sin(2*np.pi*(x[1]-0.1))/((x[0]+0.1)**3*(x[0]+x[1]-0.1))

    def G8a_g1(self, x: np.ndarray) -> np.ndarray:
        return x[0]**2 - x[1] + 1

    def G8a_g2(self, x: np.ndarray) -> np.ndarray:
        return 1 - x[0] + (x[1] - 4)**2
        # return 1 - x[0] + x[1]

    def G8b_g1(self, x: np.ndarray) -> np.ndarray:
        x[0] += 0.1
        x[1] -= 0.1
        return G8a_g1(x)
        # return x[0] - (x[1] - 0.1) + 1

    def G8b_g2(self, x: np.ndarray) -> np.ndarray:
        x[0] += 0.1
        x[1] -= 0.1
        return G8a_g2(x)
        # return 1 - x[0]
