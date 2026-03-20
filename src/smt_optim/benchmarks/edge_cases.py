import numpy as np
from .base import BenchmarkProblem



class Rosenbrock2(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 2
        self.num_fidelity = 1
        self.bounds = np.array([
            [-2, 2],
            [-2, 2]
        ])

        self.costs = [1]

        self.objective = [self.hf_objective]
        self.constraints = [
            [self.hf_constraint],
            [self.hf_constraint2]
        ]

    def hf_objective(self, x):
        res = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        return res

    # def lf_objective(self, x):
    #     res = self.hf_objective(x) + 0.1*np.sin(10*x[0] + 5*x[1])
    #     return res

    def hf_constraint(self, x):
        res = -x[0]**2 - (x[1] - 1)**1/2
        return -res

    # def lf_constraint(self, x):
    #     res = self.hf_constraint(x) - 0.1*np.sin(10*x[0] + 5*x[1])
    #     return res

    def hf_constraint2(self, x):
        # return -1 - 2*x[0] + (x[1]+15/9)**2
        # return -x[1] + 0.2 + x[0]
        return -0.5*np.sin(5*(x[0]-0.5)**2 - x[1] - np.pi/8)

    # def lf_constraint2(self, x):
    #     x[0] = x[0] + 0.1
    #     x[1] = x[1] - 1
    #     return 0.9 * self.hf_constraint2(x)



class TwoConstraints(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 2
        self.num_fidelity = 1
        self.bounds = np.array([
            [1, 4],
            [1, 4]
        ])

        self.costs = [1]

        self.objective = [self.func]
        self.constraints = [
            [self.cstr1],
            [self.cstr2]
        ]

    def func(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x[:, 0] ** 2 + x[:, 1] ** 2).ravel()

    def cstr1(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (0.15*(x[:,0]-4)**2 + 0.1*(x[:,1]-4)**2 -1).ravel()

    def cstr2(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x[:, 0] - 0.8 * x[:, 1]).ravel()