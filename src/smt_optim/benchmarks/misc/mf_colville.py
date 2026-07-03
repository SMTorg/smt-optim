from functools import partial

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class MFColville(BenchmarkProblem):
    """
    Reference:

    (with 2 fidelity levels)


    Song, X., Lv, L., Sun, W., & Zhang, J. (2019). A radial basis function-based multi-fidelity
    surrogate model: exploring correlation between high-fidelity and low-fidelity models.
    Structural & Multidisciplinary Optimization, 60(3), 965.

    Note: (page 9) the term `(x_3^2 - x_4)` should be squared

    A in [0, 1] controls the correlation between the lf and hf function.
    A=0.0 -> corr=0.0882
    A=0.2 -> corr=0.1416
    A=0.4 -> corr=0.6978
    A=0.6 -> corr=0.9521
    A=0.8 -> corr=0.9948
    A=1.0 -> corr=1.0000
    """

    def __init__(self):
        super().__init__()

        self.num_dim = 4
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1

        self.bounds = np.array(
            [
                [-10, 10],
            ]
            * self.num_dim
        )

        self.objective = [
            partial(self.func_lf, A=0.8),
            partial(self.func),
        ]

    def func(self, x):
        term1 = 100 * (x[0] ** 2 - x[1]) ** 2
        term2 = (x[0] - 1) ** 2
        term3 = (x[2] - 1) ** 2
        term4 = 90 * (x[2] ** 2 - x[3]) ** 2
        term5 = 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
        term6 = 19.8 * (x[1] - 1) * (x[3] - 1)

        return term1 + term2 + term3 + term4 + term5 + term6

    def func_lf(self, x, A):
        term1 = self.func(A**2 * x)
        x2 = x * 2
        term2 = -(A + 0.5) * (5 * x2[0] + 4 * x2[1] + 3 * x2[2] + x2[3])
        return term1 + term2


if __name__ == "__main__":
    import scipy.optimize as so

    prob = MFColville()

    x0 = np.zeros(4)

    res = so.minimize(prob.func, x0, bounds=prob.bounds, method="SLSQP")

    print(res.fun)
    print(res.x)  # solution should be at x = [1, ..., 1]

    from smt.sampling_methods import LHS

    sampler = LHS(xlimits=prob.bounds, criterion="ese", seed=0)

    x_doe = sampler(1_000)
    x_doe = np.vstack((x_doe, np.ones(4).reshape(1, -1)))
    print(x_doe)

    A = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    y_doe_hf = np.empty(x_doe.shape[0])

    for i in range(x_doe.shape[0]):
        y_doe_hf[i] = prob.func(x_doe[i, :])

    for a in A:
        func_lf = partial(prob.func_lf, A=a)

        y_doe_lf = np.empty(x_doe.shape[0])
        for i in range(x_doe.shape[0]):
            y_doe_lf[i] = func_lf(x_doe[i, :])

        corr = np.corrcoef(y_doe_hf, y_doe_lf)[0, 1]
        print(f"A={a} -> corr={corr:.4f}")
