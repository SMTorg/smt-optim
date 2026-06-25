"""
Reference:

Mainini, L., Serani, A., Rumpfkeil, M. P., Minisci, E., Quagliarella, D., Pehlivan, H., ... & Beran, P. (2022).
Analytical benchmark problems for multifidelity optimization methods. arXiv preprint arXiv:2204.07867.

(https://arxiv.org/pdf/2204.07867)

Repository:

https://gitlab.com/qudo046/avt-331-l1-benchmarks

SMT-optim implementation of the AVT311 L1 benchmark problems were adapted from the following repository:
https://gitlab.com/qudo046/avt-331-l1-benchmarks. The implementation is slightly modified to:

- follow SMT-optim benchmark problem base class,
- (when applicable) allow users to change the problem dimension.

These implementations were validated using the available data in the reference repository. Provided under
the directory `data_smt-optim` are validation data generated with SMT-optim implementations. The headers can be
interpreted as follows:

- x_i: input value
- f_i: function value (in increasing order of fidelity)
- d_i: absolute difference with original validation data (-GNU)
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


def rhs(y, m, k):
    f = np.zeros(4)

    f[0] = y[2]
    f[1] = y[3]
    f[2] = (-k[0] - k[1]) / m[0] * y[0] + k[1] / m[0] * y[1]
    f[3] = k[1] / m[1] * y[0] + (-k[0] - k[1]) / m[1] * y[1]

    return f


def rk4(y0, t0, tf, h, m, k):
    # Calculating number of time steps
    n = int((tf - t0) / h)

    # Time-march with RK4
    for i in range(n):
        k1 = h * (rhs(y0, m, k))
        k2 = h * (rhs((y0 + k1 / 2), m, k))
        k3 = h * (rhs((y0 + k2 / 2), m, k))
        k4 = h * (rhs((y0 + k3), m, k))
        kt = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        y0 = y0 + kt
        t0 = t0 + h

    return y0[0]


class MFSpring(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array([[1., 4.]] * self.num_dim)

        self.costs = [1/60, 1.]

        self.objective = [
            partial(self.func, dt=0.6),
            partial(self.func, dt=0.01),
        ]

        self.m = np.ones(2)
        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])

        self.tags = [
            "avt311",
        ]

    def func(self, x, dt=0.01):
        return rk4(self.y0, self.t0, self.tf, dt, self.m, x)


class MFMass(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array([[1., 4.]] * self.num_dim)

        self.costs = [1/60, 1.]

        self.objective = [
            partial(self.func, dt=0.6),
            partial(self.func, dt=0.01),
        ]

        self.k = np.ones(2)
        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])

        self.tags = [
            "avt311",
        ]

    def func(self, x, dt=0.01):
        return rk4(self.y0, self.t0, self.tf, dt, x, self.k)


class MFSpringMass(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 4
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1
        self.bounds = np.array([[1., 4.]] * self.num_dim)

        self.costs = [1/60, 1.]

        self.objective = [
            partial(self.func, dt=0.6),
            partial(self.func, dt=0.01),
        ]

        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])

        self.tags = [
            "avt311",
        ]

    def func(self, x, dt=0.01):
        m = x[2:]
        k = x[:2]
        return rk4(self.y0, self.t0, self.tf, dt, m, k)


if __name__ == "__main__":

    from smt_optim.benchmarks.registry import get_problem
    from smt_optim.subsolvers.multistart import multistart_minimize

    prob_setups = [
        ("Forrester", "FORRESTER", None),
        ("MFRosenbrock", "ROSENBROCK", 2),
        ("MFRosenbrock", "ROSENBROCK", 5),
        ("MFRosenbrock", "ROSENBROCK", 10),
        ("MFRastrigin", "RASTRIGIN", 2),
        ("MFRastrigin", "RASTRIGIN", 5),
        ("MFRastrigin", "RASTRIGIN", 10),
        ("Alos1", "ALOS", None),
        ("Alos", "ALOS", 2),
        ("Alos", "ALOS", 3),
        ("MFSpring", "SPRING", None),
        ("MFMass", "MASS", None),
        ("MFSpringMass", "SPRINGMASS", None),
    ]

    for setup in prob_setups:
        local_name = setup[0]
        # ref_name = setup[1]
        num_dim = setup[2]

        prob = get_problem(local_name)

        if num_dim is not None:
            prob.set_dim(num_dim)
        else:
            num_dim = prob.num_dim





    for setup in prob_setups:
        local_name = setup[0]
        # ref_name = setup[1]
        num_dim = setup[2]

        prob = get_problem(local_name)

        if num_dim is not None:
            prob.set_dim(num_dim)
        else:
            num_dim = prob.num_dim

        print(f"======= Problem: {local_name:<16} =======")

        data = np.loadtxt(f"data_smt-optim/{local_name}_d{num_dim}.txt")

        x = data[:, :num_dim]
        obj_ref = data[:, num_dim:num_dim + prob.num_fidelity]

        obj_exp = np.empty_like(obj_ref)

        for i in range(x.shape[0]):
            for lvl in range(prob.num_fidelity):
                tmp_val = prob.objective[lvl](x[i, :])
                obj_exp[i, lvl] = tmp_val.item() if isinstance(tmp_val, np.ndarray) else tmp_val

        delta = np.abs(obj_exp - obj_ref)
        max_error = np.max(delta)
        print(f"Max delta with data = {max_error:.4e}")


        def sp_objective(x):
            val = prob.objective[-1](x)
            return val.item() if isinstance(val, np.ndarray) else val


        res = multistart_minimize(sp_objective, prob.bounds)

        print(f"Solution: f = {res.fun:.3e} | x = {res.x}")

