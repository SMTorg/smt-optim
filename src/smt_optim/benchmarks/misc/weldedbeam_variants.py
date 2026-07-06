from functools import partial

import numpy as np

import smt.design_space as ds

from smt_optim.benchmarks.base import BenchmarkProblem


class MixedVarWeldedBeamDesign(BenchmarkProblem):
    """
    References:
    Datta, D., & Figueira, J. R. (2011). A real-integer-discrete-coded particle swarm optimization for design problems.
    Applied Soft Computing, 11(4), 3625-3633.
    - Tran, A., Wildey, T., & McCann, S. (2020). sMF-BO-2CoGP: A sequential multi-fidelity constrained
    Bayesian optimization framework for design applications. Journal of Computing and Information
    Science in Engineering, 20(3), 031007.
    """

    def __init__(self):
        super().__init__()

        self.name = "MixedVarWeldedBeamDesign"
        self.num_dim: int = 6
        self.num_obj = 1
        self.num_cstr = 4
        self.num_fidelity = 1

        self.bounds = None

        self.design_space = ds.DesignSpace(
            [
                ds.CategoricalVariable([0, 1]),
                ds.CategoricalVariable([0, 1, 2, 3]),
                ds.FloatVariable(0.0625, 2.0),
                ds.FloatVariable(0.1, 10.0),
                ds.FloatVariable(2.0, 20.0),
                ds.FloatVariable(0.0625, 2.0),
            ]
        )

        self.step = 0.0625

        # design space as presented in (Datta, 2011)
        self.alt_design_space = ds.DesignSpace(
            [
                ds.CategoricalVariable([0, 1]),  # w
                ds.CategoricalVariable([0, 1, 2, 3]),  # m
                ds.OrdinalVariable(
                    list(np.arange(0.0625, 2.0 + self.step, self.step))
                ),  # h
                ds.FloatVariable(0.1, 10.0),  # l
                ds.OrdinalVariable(
                    list(np.arange(2.0, 20.0 + self.step, self.step))
                ),  # t
                ds.OrdinalVariable(
                    list(np.arange(0.0625, 2.0 + self.step, self.step))
                ),  # b
            ]
        )

        # reference test function -> h, t and b are discrete

        self.tags = []

        self.F = 6_000
        self.L = 14
        self.delta_max = 0.25

        self.C1 = np.array([0.1047, 0.0489, 0.5235, 0.5584])
        self.C2 = np.array([0.0481, 0.0224, 0.2405, 0.2566])
        self.sigma_d = np.array([30e3, 8e3, 5e3, 8e3])
        self.E = np.array([30e6, 14e6, 10e6, 16e6])
        self.G = np.array([12e6, 6e6, 4e6, 6e6])

        self.constraints = [
            self.shear_stress,
            self.bending_stress,
            # self.buckling,
            self.deflection,
            self.side_constraints,
        ]

    def objective(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)
        return (1 + self.C1[m]) * (w * t + l) * h**2 + self.C2[m] * t * b * (self.L + l)

    def weld_properties(self, x):

        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)

        if w == 0:
            A = np.sqrt(2) * h * l
            J = np.sqrt(2) * h * l * ((h + t) ** 2 / 4 + l**2 / 12)
            R = 1 / 2 * np.sqrt(l**2 + (h + t) ** 2)
            cos_theta = l / (2 * R)
            return A, J, R, cos_theta

        # w == 1
        else:
            A = np.sqrt(2) * h * (t + l)
            J = np.sqrt(2) * h * l * ((h + t) ** 2 / 4 + l**2 / 12) + np.sqrt(
                2
            ) * h * t * ((h + l) ** 2 / 4 + t**2 / 12)
            R = max(
                1 / 2 * np.sqrt(l**2 + (h + t) ** 2),
                1 / 2 * np.sqrt(t**2 + (h + l) ** 2),
            )
            cos_theta = l / (2 * R) if l < t else t / (2 * R)
            return A, J, R, cos_theta

    def sigma(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        return 6 * self.F * self.L / (t**2 * b)

    def delta(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)
        return 4 * self.F * self.L**3 / (self.E[m] * t**3 * b)

    def P_c(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)
        return (
            (4.013 * t * b**3 * np.sqrt(self.E[m] * self.G[m]))
            / (6 * self.L**2)
            * (1 - t / (4 * self.L) * np.sqrt(self.E[m] / self.G[m]))
        )

    def tau(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        A, J, R, cos_theta = self.weld_properties(x)
        tau_p = self.F / A
        tau_pp = (self.F * (self.L + l / 2) * R) / J
        return np.sqrt(tau_p**2 + tau_pp**2 + 2 * tau_p * tau_pp * cos_theta)

    def shear_stress(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)
        return self.tau(x) - 0.577 * self.sigma_d[m]

    def bending_stress(self, x):
        w, m, h, l, t, b = x  # noqa: E741
        w, m = int(w), int(m)
        return self.sigma(x) - self.sigma_d[m]

    def buckling(self, x):
        # not a constraint in reference test problem
        w, m, h, l, t, b = x  # noqa: E741
        return h - b

    def deflection(self, x):
        return self.F - self.P_c(x)

    def side_constraints(self, x):
        return self.delta(x) - self.delta_max


class MFWeldedBeamDesign(BenchmarkProblem):
    """
    References:

    Datta, D., & Figueira, J. R. (2011). A real-integer-discrete-coded particle swarm optimization for
    design problems. Applied Soft Computing, 11(4), 3625-3633.

    Tran, A., Wildey, T., & McCann, S. (2020). sMF-BO-2CoGP: A sequential multi-fidelity constrained
    Bayesian optimization framework for design applications. Journal of Computing and Information
    Science in Engineering, 20(3), 031007.
    """

    def __init__(self):
        super().__init__()

        self.name = "MFWeldedBeamDesign"

        self.ref_problem = MixedVarWeldedBeamDesign()

        self.num_dim: int = 4
        self.num_obj = 1
        self.num_cstr = 5
        self.num_fidelity = 4

        self.bounds = np.array(
            [
                [0.0625, 2],
                [0.1, 10],
                [2.0, 20],
                [0.0625, 2],
            ]
        )

        self.fidelities = [
            (0, 3),
            (0, 2),
            (0, 1),
            (0, 0),
        ]

        self.objective = []

        self.constraints = [
            [],
            [],
            [],
            [],
            [],
        ]

        for w, m in self.fidelities:
            self.objective.append(
                partial(self.expand_input, w=w, m=m, func=self.ref_problem.objective)
            )

            self.constraints[0].append(
                partial(self.expand_input, w=w, m=m, func=self.ref_problem.shear_stress)
            )

            self.constraints[1].append(
                partial(
                    self.expand_input, w=w, m=m, func=self.ref_problem.bending_stress
                )
            )

            self.constraints[2].append(
                partial(self.expand_input, w=w, m=m, func=self.ref_problem.buckling)
            )

            self.constraints[3].append(
                partial(self.expand_input, w=w, m=m, func=self.ref_problem.deflection)
            )

            self.constraints[4].append(
                partial(
                    self.expand_input, w=w, m=m, func=self.ref_problem.side_constraints
                )
            )

    def expand_input(self, x, w, m, func):
        w, m = int(w), int(m)
        h, l, t, b = x  # noqa: E741
        x_exp = np.array([w, m, h, l, t, b])
        return func(x_exp)


if __name__ == "__main__":
    # Reference: Datta, D., & Figueira, J. R. (2011). A real-integer-discrete-coded particle swarm optimization
    # for design problems. Applied Soft Computing, 11(4), 3625-3633.

    problem = MixedVarWeldedBeamDesign()

    # w, m, h, l, t, b = x
    x1 = np.array([1, 0, 0.1875, 1.6849, 8.2500, 0.2500])
    x2 = np.array([1, 0, 0.1250, 4.115994, 8.3125, 0.2500])
    x3 = np.array([1, 0, 0.1875, 1.782103, 8.2500, 0.2500])

    # obj, c1, c2, c3, c4
    # obj, g4, g1, g2, g3
    sol1 = np.array([1.9422, 394.196461, -380.165289, -402.047213, -0.234362])
    sol2 = np.array([2.025363, -363.700864, -823.901860, -435.707814, -0.234712])
    sol3 = np.array([1.955301, -0.004258, -380.165289, -402.047213, -0.234362])

    pairs = [
        (x1, sol1),
        (x2, sol2),
        (x3, sol3),
    ]

    for x_ref, sol_ref in pairs:
        vals = np.empty(1 + problem.num_cstr)

        vals[0] = problem.objective(x_ref)

        for c_idx in range(problem.num_cstr):
            vals[1 + c_idx] = problem.constraints[c_idx](x_ref)

        print(f"L2 error = {np.linalg.norm(vals - sol_ref):.4e}")
        # print(f"refs={sol_ref}")
        # print(f"vals={vals}")

    # Reference:
    # Tran, A., Tran, M., & Wang, Y. (2019). Constrained mixed-integer Gaussian mixture Bayesian optimization
    # and its applications in designing fractal and auxetic metamaterials. Structural and Multidisciplinary
    # Optimization, 59(6), 2131-2154.

    # reference paper specifies 3 solutions
    # none of the solutions are feasible
    # x1 matches reference objective
    # x2 and x3 are the same vectors (typo in ref. paper)
    # x2 and x3 matches other specified objective value: 1.6297

    x1 = np.array([0, 0, 0.24920115, 5.30060037, 7.12520087, 0.25345267])
    sol1 = 2.04016262

    x2 = np.array([1, 0, 0.16934934, 5.61720010, 4.90884889, 0.27985016])
    sol2 = 1.68206763

    x3 = np.array([1, 0, 0.16934934, 5.61720010, 4.90884889, 0.27985016])
    sol3 = 1.66457625

    pairs = [
        (x1, sol1),
        (x2, sol2),
        (x3, sol3),
    ]

    for x_ref, sol_ref in pairs:
        vals = np.empty(1 + problem.num_cstr)

        vals[0] = problem.objective(x_ref)

        for c_idx in range(problem.num_cstr):
            vals[1 + c_idx] = problem.constraints[c_idx](x_ref)

        print(f"L2 error = {np.linalg.norm(vals[0] - sol_ref):.4e}")
        # print(vals[0], sol_ref)

    # Reference: Tran, A., Wildey, T., & McCann, S. (2020). sMF-BO-2CoGP: A sequential multi-fidelity constrained
    # Bayesian optimization framework for design applications. Journal of Computing and Information Science in
    # Engineering, 20(3), 031007.

    problem = MFWeldedBeamDesign()

    x0 = np.full(4, 0.5)

    for lvl in range(problem.num_fidelity):
        vals = np.empty(5)

        vals[0] = problem.objective[lvl](x0)

        c_vals = []
        for c_idx in range(problem.num_cstr):
            vals[c_idx] = problem.constraints[c_idx][lvl](x0)

        print(f"{vals}")
