from functools import partial
import warnings

import numpy as np

from smt_optim.benchmarks.base import BenchmarkProblem


class MFBorehole(BenchmarkProblem):
    """
    Reference:

    (with 2 fidelity levels)
    Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and analysis of high-accuracy and low-accuracy
    computer codes. Technometrics, 55(1), 37-46.

    (with 3 fidelity levels)
    Tran, A., Wildey, T., & McCann, S. (2020). sMF-BO-2CoGP: A sequential multi-fidelity constrained Bayesian
    optimization framework for design applications. Journal of Computing and Information Science in Engineering,
    20(3), 031007.

    """

    def __init__(self):
        super().__init__()

        self.num_dim = 8
        self.num_cstr = 0
        self.num_fidelity = 2
        self.num_obj = 1

        self.bounds = np.array(
            [
                [0.05, 0.15],
                [100, 50_000],
                [63_070, 115_600],
                [990, 1_110],
                [63.1, 116],
                [700, 820],
                [1_120, 1680],
                [9_855, 12_045],
            ]
        )

        self.objective = [
            partial(self.func, A=1.5),
            partial(self.func, A=1.0),
        ]

    def set_num_fid(self, num_fidelity):

        if num_fidelity not in [2, 3]:
            warnings.warn(
                "MF Borehole can have 2 or 3 fidelity levels. Defaulting to 2."
            )
            num_fidelity = 2

        if num_fidelity == 2:
            self.objective = [
                partial(self.func, A=5, B=1.5),
                partial(self.func, A=2 * np.pi, B=1.0),
            ]

        else:
            self.objective = [
                partial(self.func, A=7, B=0.5),
                partial(self.func, A=5, B=1.5),
                partial(self.func, A=2 * np.pi, B=1.0),
            ]

        self.num_fidelity = len(self.objective)

    def func(self, x, A=2 * np.pi, B=1.0):

        term1 = A * x[2] * (x[3] - x[5])
        term2 = np.log10(x[1] / x[0])
        term3 = (
            B
            + (2 * x[6] * x[2]) / (np.log(x[1] / x[0]) * x[0] ** 2 * x[7])
            + x[2] / x[4]
        )

        return term1 / (term2 * term3)


if __name__ == "__main__":
    import scipy.optimize as so

    prob = MFBorehole()

    x0 = np.array([7.0e-02, 2.0e02, 7.0e04, 1.0e03, 1.0e02, 7.5e02, 1.4e03, 1.0e04])
    x_ref = np.array(
        [
            5.00000000e-02,
            2.00184597e02,
            6.99999998e04,
            9.90000000e02,
            9.69866639e01,
            8.20000000e02,
            1.68000000e03,
            9.96047429e03,
        ]
    )

    res = so.minimize(prob.objective[-1], x0, bounds=prob.bounds, method="SLSQP")

    # print(res.fun)
    # print(res.x)
    print(np.linalg.norm(res.x - x_ref))

    prob.set_num_fid(2)
    print(len(prob.objective))

    prob.set_num_fid(3)
    print(len(prob.objective))
