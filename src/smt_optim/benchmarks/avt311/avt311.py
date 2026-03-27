"""
Paper:  https://arxiv.org/pdf/2204.07867
Code:   https://gitlab.com/qudo046/avt-331-l1-benchmarks
"""
from functools import partial
import math

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class Alos1(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 1
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([
            [0, 1],
        ])

        # self.costs = [0.15/9, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]


    def f(self, x, fid=1):
        if fid == 1:
            return np.sin(30.0 * (x - 0.9) ** 4) * np.cos(2.0 * (x - 0.9)) + (x - 0.9) / 2.0
        else:
            return (self.f(x) - 1.0 + x) / (1.0 + 0.25 * x)


class Alos2(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([
            [0, 1],
            [0, 1],
        ])

        # self.costs = [0.15/9, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

    def f(self, x, fid=1):
        if fid == 1:
            return math.sin(21.0*(x[0]-0.9)**4)*math.cos(2.0*(x[0]-0.9))+(x[0]-0.7)/2.0+ 2.0*x[1]**2*math.sin(x[0]*x[1])
        else:
            return (self.f(x)-2.0+x[0]+x[1])/(5.0+0.25*x[0]+0.5*x[1])


class Alos3(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 3
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([[0, 1]] * self.num_dim)

        # self.costs = [0.15/9, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

    def f(self, x, fid=1):
        if fid == 1:
            return math.sin(21.0 * (x[0] - 0.9) ** 4) * math.cos(2.0 * (x[0] - 0.9)) + (x[0] - 0.7) / 2.0 + 2.0 * x[1] ** 2 * math.sin(x[0] * x[1]) + 3.0 * x[2] ** 3 * math.sin(x[0] * x[1] * x[2])
        else:
            return (f(x) - 2.0 + x[0] + x[1] + x[2]) / (5.0 + 0.25 * x[0] + 0.5 * x[1] - 0.75 * x[2])


class MFRosenbrock2D(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 3
        self.bounds = np.array([[-2, 2]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
            partial(self.f, fid=2),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

    def f(self, x, fid=2):

        f = 0

        # high fidelity
        if fid == 2:
            for k in range(self.num_dim - 1):
                f += 100.0 * (x[k + 1] - x[k] ** 2) ** 2 + (1.0 - x[k]) ** 2

        # medium fidelity
        elif fid == 1:
            for k in range(self.num_dim - 1):
                f += 50.0 * (x[k + 1] - x[k] ** 2) ** 2 + (-2.0 - x[k]) ** 2 - 0.5 * x[k]
            f -= 0.5 * x[self.num_dim - 1]

        # low fidelity
        if fid == 0:
            for k in range(self.num_dim - 1):
                f += 100.0 * (x[k + 1] - x[k] ** 2) ** 2 + (1.0 - x[k]) ** 2

            b0 = 4.0
            bi = [0.5 for x in range(self.num_dim)]
            a0 = 10.0
            ai = [0.25 for x in range(self.num_dim)]  # Be careful that the denom != 0
            # subtract additive terms
            delta = b0
            for i in range(self.num_dim):
                delta = delta + bi[i] * x[i]
            uval = f - delta
            # divide by multiplicative terms
            fac = a0
            for i in range(self.num_dim):
                fac = fac + ai[i] * x[i]
            uval = uval / fac

            f = uval

        return f


class MFRastrigin2D(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 3
        self.bounds = np.array([[-0.1, 0.2]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
            partial(self.f, fid=2),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

        self.xStar = [0.1 for x in range(self.num_dim)]
        self.Xtrasla = [0.0 for x in range(self.num_dim)]
        self.Theta = 0.2
        self.Rmat = [[math.cos(self.Theta), -math.sin(self.Theta)], [math.sin(self.Theta), math.cos(self.Theta)]]

    def f(self, x, fid=2):

        for k in range(self.num_dim):
            self.Xtrasla[k] = x[k] - self.xStar[k]

        z = [0.0 for x in range(self.num_dim)]
        for k in range(self.num_dim):
            for j in range(self.num_dim):
                z[k] += self.Rmat[k][j] * self.Xtrasla[j]

        f = 0.0
        for k in range(self.num_dim):
            f += (z[k] ** 2.0 + 1.0 - math.cos(10.0 * math.pi * z[k]))

        if fid == 2:
            Phi = 10000.0
        elif fid == 1:
            Phi = 5000.0
        else:
            Phi = 2500.0

        TH = 1.0 - 0.0001 * Phi
        a = TH
        w = 10.0 * math.pi * TH
        b = 0.5 * math.pi * TH

        er = 0.0
        for k in range(self.num_dim):
            er += a * (math.cos(w * z[k] + b + math.pi) ** 2.0)

        f += er

        return f


class Forrester(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 1        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 4
        self.bounds = np.array([[0, 1]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
            partial(self.f, fid=2),
            partial(self.f, fid=3),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

    def f(self, x, fid=3):

        if fid == 3:
            f = ((6.0 * x - 2.0) ** 2.0) * math.sin(12.0 * x - 4.0)
        elif fid == 2:
            f = ((5.50 * x - 2.5) ** 2.0) * math.sin(12.0 * x - 4.0)
        elif fid == 1:
            f = 0.75 * self.f(x) + 5.0 * (x - 0.5) - 2.0
        else:
            f = 0.5 * self.f(x) + 10.0 * (x - 0.5) - 5.0

        return f


class DiscForrester(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 1        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([[0, 1]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objectives = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        self.constraints = []

        self.tags = [
            "avt311"
        ]

    def f(self, x, fid=1):

        if x <= 0.5:
            f = ((6.0 * x - 2.0) ** 2.0) * math.sin(12.0 * x - 4.0)
        elif x > 0.5:
            f = 10 + ((6.0 * x - 2.0) ** 2.0) * math.sin(12.0 * x - 4.0)

        if fid == 0:
            if x <= 0.5:
                f = 0.5 * f + 10.0 * (x - 0.5) - 5.0
            elif x > 0.5:
                f = 0.5 * f + 10.0 * (x - 0.5) - 7.0

        return f


