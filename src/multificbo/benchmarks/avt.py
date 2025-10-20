import numpy as np
from .base import BenchmarkProblem


# rhs for first-order system of ODEs in spring-mass system
def f(t, y, m, k):
    f = np.array([0.0, 0.0, 0.0, 0.0])

    f[0] = y[2]
    f[1] = y[3]
    f[2] = (-k[0] - k[1]) / m[0] * y[0] + k[1] / m[0] * y[1]
    f[3] = k[1] / m[1] * y[0] + (-k[0] - k[1]) / m[1] * y[1]

    return f


# RK-4 method
def rk4(y0, t0, tf, h, m, k):
    # Calculating number of time steps
    n = int((tf - t0) / h)

    # Time-march with RK4
    for i in range(n):
        k1 = h * (f(t0, y0, m, k))
        k2 = h * (f((t0 + h / 2), (y0 + k1 / 2), m, k))
        k3 = h * (f((t0 + h / 2), (y0 + k2 / 2), m, k))
        k4 = h * (f((t0 + h), (y0 + k3), m, k))
        kt = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        y0 = y0 + kt
        t0 = t0 + h

    return y0[0]


class Mass(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([
            [1, 4],
            [1, 4]
        ])

        self.costs = []

        obj_hf = lambda x, dt=0.01: self.mass(x, dt)
        obj_lf = lambda x, dt=0.6: self.mass(x, dt)

        self.objective = [obj_lf, obj_hf]
        self.constraints = []

        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.k = np.array([1.0, 1.0])

    def mass(self, x: np.ndarray, dt: float):
        f = rk4(self.y0, self.t0, self.tf, dt, x, self.k)
        return f

class Spring(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([
            [1, 4],
            [1, 4]
        ])

        self.costs = []

        obj_hf = lambda x, dt=0.01: self.spring(x, dt)
        obj_lf = lambda x, dt=0.6: self.spring(x, dt)

        self.objective = [obj_lf, obj_hf]
        self.constraints = []

        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.m = np.array([1.0, 1.0])

    def spring(self, x: np.ndarray, dt: float):
        f = rk4(self.y0, self.t0, self.tf, dt, self.m, x)
        return f


class SpringMass(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 4
        self.num_cstr = 0
        self.num_fidelity = 2
        self.bounds = np.array([
            [1, 4],
            [1, 4]
        ])

        self.costs = []

        obj_hf = lambda x, dt=0.01: self.spring(x, dt)
        obj_lf = lambda x, dt=0.6: self.sping(x, dt)

        self.objective = [obj_lf, obj_hf]
        self.constraints = []

        self.t0 = 0.0
        self.tf = 6.0
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0])

    def spring_mass(self, x: np.ndarray, dt: float):

        m = x[2:4]
        k = x[0:2]

        f = rk4(self.y0, self.t0, self.tf, dt, m, k)

        return f
