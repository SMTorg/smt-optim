"""
Paper:  https://arxiv.org/pdf/2204.07867
Code:   https://gitlab.com/qudo046/avt-331-l1-benchmarks
"""
from functools import partial
import warnings

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class Alos1(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 1
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array([
            [0, 1],
        ])

        # self.costs = [0.15/9, 1]

        self.objective = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
        ]

    def f(self, x, fid=1):
        if fid == 1:
            return np.sin(30.0 * (x - 0.9) ** 4) * np.cos(2.0 * (x - 0.9)) + (x - 0.9) / 2.0
        else:
            return (self.f(x) - 1.0 + x) / (1.0 + 0.25 * x)


class Alos(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "Alos"
        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array([
            [0, 1],
            [0, 1],
        ])

        # self.costs = [0.15/9, 1]

        self.objective = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
            "n_variable",
        ]


    def set_dim(self, dim: int):
        if "n_variable" in self.tags:

            if dim < 2 or dim > 3:
                warnings.warn("Alos is either a 2D or 3D benchmark problem.")

            self.num_dim = dim
            self.bounds = self.bounds[-1, :].reshape(1, 2)
            self.bounds = self.bounds.repeat(dim, axis=0)


    def f(self, x, fid=1):
        if fid == 1:

            val = np.sin(21*(x[0] - 0.9)**4) * np.cos(2*(x[0] - 0.9)) + (x[0] - 0.7)/2

            for i in range(1, self.num_dim):
                prod = np.prod(x[:i+1])
                val += (i+1) * x[i]**(i+1) * np.sin(prod)

            return val

        else:

            val = self.f(x, fid=1)

            num = val - 2 + np.sum(x)

            term1 = 0.
            for i in range(0, 2):
                term1 += (i+1)*x[i]
            term1 *= 0.25

            term2 = 0.
            for i in range(2, self.num_dim):
                term2 += (i+1)*x[i]
            term2 *= 0.25

            denom = 5. + term1 - term2

            return num/denom


class MFRosenbrock(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 3
        self.num_obj = 1
        self.bounds = np.array([[-2, 2]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objective = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
            partial(self.f, fid=2),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
            "n_variable",
        ]

    def f(self, x, fid=2):

        val = 0.

        if fid == 2:
            for i in range(self.num_dim-1):
                val += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

        elif fid == 1:
            for i in range(self.num_dim-1):
                val += 50*(x[i+1] - x[i]**2)**2 + (-2 - x[i])**2

            val -= 0.5*np.sum(x)

        elif fid == 0:
            sum_x = np.sum(x)
            val = (self.f(x, fid=2) - 4. - 0.5*sum_x)/(10 + 0.25*sum_x)

        else:
            raise ValueError()

        return val


class MFRastrigin(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 3
        self.num_obj = 1
        self.bounds = np.array([[-0.1, 0.2]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objective = [
            partial(self.fi, phi=2_500),
            partial(self.fi, phi=5_000),
            partial(self.fi, phi=10_000),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
            "n_variable",
        ]

        self.xStar = np.full(self.num_dim, 0.1)
        self.theta = 0.2
        self.Rmat = self.rotation_matrix(self.num_dim,
                                         np.zeros((self.num_dim, self.num_dim-1)),
                                         self.theta)



    def set_dim(self, dim: int):
        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds = self.bounds[-1, :].reshape(1, 2)
            self.bounds = self.bounds.repeat(dim, axis=0)

            self.xStar = np.full(self.num_dim, 0.1)
            self.Rmat = self.rotation_matrix(self.num_dim,
                                             np.zeros((self.num_dim, self.num_dim-1)),
                                             self.theta)


    def f1(self, z):
        return np.sum(z**2 + 1 - np.cos(10*np.pi*z))


    def z(self, x):
        return self.Rmat @ (x - self.xStar)


    def resolution_error(self, z: np.ndarray, phi: float):

        omega = 1 - phi/10_000
        a = omega
        w = 10*np.pi*omega
        b = 0.5*np.pi*omega

        return np.sum(a * np.cos(w*z + b + np.pi)**2)

    def rotation_matrix(self, n, v, theta):
        """
        Aguilera-Perez algorithm

        Parameters
        ----------
        n : int
            Dimension
        v : (n, n-1) array
            Input matrix
        theta : float
            Final rotation angle

        Returns
        -------
        R : (n, n) array
            Final rotation matrix
        """

        v = v.copy().astype(float)
        M = np.eye(n)

        for c in range(n - 2):
            for rr in range(n - 1, c, -1):
                t = np.arctan2(v[rr, c], v[rr - 1, c])

                R = np.eye(n)

                # Givens rotation in (rr-1, rr)
                coss = np.cos(t)
                sins = np.sin(t)

                R[rr, rr] = coss
                R[rr, rr - 1] = sins
                R[rr - 1, rr] = -sins
                R[rr - 1, rr - 1] = coss

                # v = R v
                v1 = R @ v
                v = v1

                # M = R M
                M1 = R @ M
                M = M1

        R = np.eye(n)

        coss = np.cos(theta)
        sins = np.sin(theta)

        R[n - 1, n - 1] = coss
        R[n - 1, n - 2] = sins
        R[n - 2, n - 1] = -sins
        R[n - 2, n - 2] = coss

        B = R @ M
        X = np.linalg.solve(M, B)

        return X

    def fi(self, x: np.ndarray, phi: float):
        z = self.z(x)
        return self.f1(z) + self.resolution_error(z, phi)


class Forrester(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 1        # could be variable with d -> [4, 7]
        self.num_cstr = 0
        self.num_fidelity = 4
        self.num_obj = 1
        self.bounds = np.array([[0, 1]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objective = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
            partial(self.f, fid=2),
            partial(self.f, fid=3),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
        ]

    def f(self, x, fid=3):

        if fid == 3:
            f = ((6.0 * x - 2.0) ** 2.0) * np.sin(12.0 * x - 4.0)
        elif fid == 2:
            f = ((5.50 * x - 2.5) ** 2.0) * np.sin(12.0 * x - 4.0)
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
        self.num_obj = 1
        self.bounds = np.array([[0, 1]] * self.num_dim)

        # self.costs = [0.1, 1]

        self.objective = [
            partial(self.f, fid=0),
            partial(self.f, fid=1),
        ]

        # self.constraints = []

        self.tags = [
            "avt311",
        ]

    def f(self, x, fid=1):

        if x <= 0.5:
            f = ((6.0 * x - 2.0) ** 2.0) * np.sin(12.0 * x - 4.0)
        elif x > 0.5:
            f = 10 + ((6.0 * x - 2.0) ** 2.0) * np.sin(12.0 * x - 4.0)

        if fid == 0:
            if x <= 0.5:
                f = 0.5 * f + 10.0 * (x - 0.5) - 5.0
            elif x > 0.5:
                f = 0.5 * f + 10.0 * (x - 0.5) - 7.0

        return f


