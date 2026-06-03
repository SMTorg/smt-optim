"""
Reference: https://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page422.htm
"""


import numpy as np
from smt_optim.benchmarks.base import BenchmarkProblem

class G01(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G01"
        self.num_dim: int = 13
        self.num_obj = 1
        self.num_cstr = 9
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 100],
            [0, 100],
            [0, 100],
            [0, 1],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
            self.g7,
            self.g8,
            self.g9,
        ]

        # x* = (1,1,...,1,3,3,3,1), f(x*) = -15


    def objective(self, x):
        x1 = x[0:5]
        x2 = x[4:14]
        return 5 * np.sum(x1) - 5 * np.sum(x1 ** 2) - np.sum(x2)

    def g1(self, x):
        return  2 * x[0] + 2 * x[1] + x[9] + x[10] - 10

    def g2(self, x):
        return 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10

    def g3(self, x):
        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10

    def g4(self, x):
        return -8 * x[0] + x[9]

    def g5(self, x):
        return -8 * x[1] + x[10]

    def g6(self, x):
        return -8 * x[2] + x[11]

    def g7(self, x):
        return -2 * x[3] - x[4] + x[9]

    def g8(self, x):
        return -2 * x[5] - x[6] + x[10]

    def g9(self, x):
        return -2 * x[7] - x[8] + x[11]


class G02(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G02"
        self.num_dim: int | str = 20
        self.num_obj = 1
        self.num_cstr = 2
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 10],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
            "n_variable",
        ]

        self.constraints = [
            self.g1,
            self.g2,
        ]

        # n = 20,  f(x*) = 0.803619. Have not found this solution yet
        # x_star = np.array([3.162460616, 3.128331428, 3.094792129, 3.061451495, 3.027929159,2.993826067,
        #     2.958668718, 2.921842273, 0.494825115, 0.488357110,0.482316427, 0.476645751, 0.471295508, 0.466230993,
        #     0.461420050,0.456836648, 0.452458769, 0.448267622, 0.444247010, 0.440382860])


    def objective(self, x):
        cos_x = np.cos(x)
        sum_jx = np.sum(np.linspace(1, self.num_dim, self.num_dim) * x ** 2)
        return -np.abs((np.sum(cos_x ** 4) - 2 * np.prod(cos_x ** 2)) / np.sqrt(sum_jx))

    def g1(self, x):
        return -np.prod(x) + 0.75

    def g2(self, x):
        return np.sum(x) - 7.5 * self.num_dim


class G03(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G03"
        self.num_dim: int | str = 2
        self.num_obj = 1
        self.num_cstr = 1
        self.num_fidelity = 1

        self.cstr_types = ["eq"]

        self.bounds = np.array([
            [0, 1],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
            "n_variable",
        ]

        self.constraints = [
            self.h1,            # equality constraint
        ]

        # x* =  (1/n^0.5, …, 1/n^0.5), f(x*) = 1

    def objective(self, x):
        cst = np.sqrt(self.num_dim) ** self.num_dim
        return -cst * np.prod(x)

    def h1(self, x):
        return np.sum(x ** 2) - 1


class G04(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G04"
        self.num_dim: int = 5
        self.num_obj = 1
        self.num_cstr = 6
        self.num_fidelity = 1

        self.bounds = np.array([
            [78, 102],
            [33, 45],
            [27, 45],
            [27, 45],
            [27, 45],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
        ]

        # x* =  (78,33,29.995,45,36.7758), f(x*) = -30665.539


    def objective(self, x):
        return 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141

    def u(self, x):
        return 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]

    def v(self, x):
        return 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2

    def w(self, x):
        return 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

    def g1(self, x):
        return -self.u(x)

    def g2(self, x):
        return self.u(x) - 92

    def g3(self, x):
        return -self.v(x) + 90

    def g4(self, x):
        return self.v(x) - 110

    def g5(self, x):
        return -self.w(x) + 20

    def g6(self, x):
        return self.w(x) - 25


class G05(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G05"
        self.num_dim: int = 4
        self.num_obj = 1
        self.num_cstr = 5
        self.num_fidelity = 1

        self.cstr_types = [
            "ineq",
            "ineq",
            "eq",
            "eq",
            "eq",
        ]

        self.bounds = np.array([
            [0, 1200],
            [0, 1200],
            [-0.55, 0.55],
            [-0.55, 0.55],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.h1,
            self.h2,
            self.h3,
        ]

        # x* =  (679.9453,1026,0.118876,-0.3962336), f(x*) = 5126.4981


    def objective(self, x):
        return 3*x[0] + 1e-6*x[0]**3 + 2*x[1] + 2e-6/3*x[1]**3

    def g1(self, x):
        return x[2]-x[3] - 0.55

    def g2(self, x):
        return x[3] - x[2] - 0.55

    def h1(self, x):
        return np.abs(1000*(np.sin(-x[2] - 0.25) + np.sin(-x[3] - 0.25)) + 894.8 - x[0])

    def h2(self, x):
        return np.abs(1000 * (np.sin(x[2] - 0.25) + np.sin(x[2] - x[3] - 0.25)) + 894.8 - x[1])

    def h3(self, x):
        return np.abs(1000*(np.sin(x[3] - 0.25) + np.sin(x[3] - x[2] - 0.25)) + 1294.8)


class G06(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G06"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 2
        self.num_fidelity = 1

        self.bounds = np.array([
            [13, 100],
            [0, 100],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
        ]

        # x* =  (14.095,0.84296), f(x*) = -6961.81388


    def objective(self, x):
        return (x[0] - 10)**3 + (x[1] - 20)**3

    def g1(self, x):
        return -(x[0] - 5)**2 - (x[1] - 5)**2 + 100

    def g2(self, x):
        return (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81


class G07(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G07"
        self.num_dim: int = 10
        self.num_obj = 1
        self.num_cstr = 8
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
            self.g7,
            self.g8,
        ]

        # x*=(2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574,1.321644, 9.828726, 8.280092, 8.375927),
        # f(x*) = 24.3062091

    def objective(self, x):
        return x[0]**2+x[1]**2+x[0]*x[1]-14*x[0]-16*x[1]+(x[2]-10)**2 + 4*(x[3]-5)**2+(x[4]-3)**2+2*(x[5]-1)**2+5*x[6]**2 + 7*(x[7]-11)**2+2*(x[8]-10)**2+(x[9]-7)**2+45

    def g1(self, x):
        return 4*x[0]+5*x[1]-3*x[6]+9*x[7]-105

    def g2(self, x):
        return 10*x[0]-8*x[1]-17*x[6]+2*x[7]

    def g3(self, x):
        return -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12

    def g4(self, x):
        return 3*(x[0]-2)**2+4*(x[1]-3)**2+2*x[2]**2-7*x[3]-120

    def g5(self, x):
        return 5*x[0]**2+8*x[1]+(x[2]-6)**2-2*x[3]-40

    def g6(self, x):
        return 0.5*(x[0]-8)**2+2*(x[1]-4)**2+3*x[4]**2-x[5]-30

    def g7(self, x):
        return x[0]**2+2*(x[1]-2)**2-2*x[0]*x[1]+14*x[4]-6*x[5]

    def g8(self, x):
        return -3*x[0]+6*x[1]+12*(x[8]-8)**2-7*x[9]


class G08(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G08"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 2
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 10],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
        ]

        # x* = (1.2279713, 4.2453733) , f (x*) = 0.095825


    def objective(self, x):
        return -(np.sin(2*np.pi*x[0])**3 * np.sin(2*np.pi*x[1]))/(x[0]**3 * (x[0] + x[1]))

    def g1(self, x):
        return x[0]**2 - x[1] + 1

    def g2(self, x):
        return 1 - x[0] + (x[1] - 4)**2


class G09(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G09"
        self.num_dim: int = 7
        self.num_obj = 1
        self.num_cstr = 4
        self.num_fidelity = 1

        self.bounds = np.array([
            [-10, 10],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
        ]

        # x* = (2.330499, 1.951372,−0.4775414, 4.365726,−0.6244870, 1.038131,1.594227), f (x*) = 680.6300573


    def objective(self, x):
        return (x[0]-10)**2+5*(x[1]-12)**2+x[2]**4+3*(x[3]-11)**2 + 10*x[4]**6+7*x[5]**2+x[6]**4-4*x[5]*x[6]-10*x[5]-8*x[6]

    def v1(self, x):
        return 2*x[0]**2

    def v2(self, x):
        return x[1]**2

    def g1(self, x):
        return self.v1(x)+3*self.v2(x)**2+x[2]+4*x[3]**2+5*x[4]-127

    def g2(self, x):
        return 7*x[0]+3*x[1]+10*x[2]**2+x[3]-x[4]-282

    def g3(self, x):
        return 23*x[0]+self.v2(x)+6*x[5]**2-8*x[6]-196

    def g4(self, x):
        return 2*self.v1(x)+self.v2(x)-3*x[0]*x[1]+2*x[2]**2+5*x[5]-11*x[6]


class G10(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G10"
        self.num_dim: int = 8
        self.num_obj = 1
        self.num_cstr = 6
        self.num_fidelity = 1

        self.bounds = np.array([
            [100, 10_000],
            [1_000, 10_000],
            [1_000, 10_000],
            [10, 1000],
            [10, 1000],
            [10, 1000],
            [10, 1000],
            [10, 1000],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
        ]

        # x* = (579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162,395.5979), f (x*) = 7049.3307


    def objective(self, x):
        return  x[0]+x[1]+x[2]

    def g1(self, x):
        return -1+0.0025*(x[3]+x[5])

    def g2(self, x):
        return -1+0.0025*(-x[3]+x[4]+x[6])

    def g3(self, x):
        return -1+0.01*(-x[4]+x[7])

    def g4(self, x):
        return 100*x[0]-x[0]*x[5]+833.33252*x[3]-83333.333

    def g5(self, x):
        return x[1]*x[3]-x[1]*x[6]-1250*x[3]+1250*x[4]

    def g6(self, x):
        return x[2]*x[4]-x[2]*x[7]-2500*x[4]+1250000


class G11(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G11"
        self.num_dim: int = 2
        self.num_obj = 1
        self.num_cstr = 1
        self.num_fidelity = 1

        self.cstr_types = ["eq"]

        self.bounds = np.array([
            [-1, 1],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.h1,
        ]

        # x* =  ±(1/(2^0.5), 1/2), f(x*) = 0.75


    def objective(self, x):
        return  x[0]**2 + (x[1] - 1)**2

    def h1(self, x):
        return x[1] - x[0]**2


class G12(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G12"
        self.num_dim: int = 3
        self.num_obj = 1
        self.num_cstr = 1
        self.num_fidelity = 1

        self.bounds = np.array([
            [0, 10],
        ] * self.num_dim)

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
        ]

        # x* = (5, 5, 5), f(x*) = 1


    def objective(self, x):
        return -(1 - 0.01*((x[0] - 5)**2 + (x[1] - 5)**2 + (x[2] - 5)**2))

    def g1(self, x):

        z = np.empty((9, 9, 9))

        for i in range(9):
            for j in range(9):
                for k in range(9):
                    p = i+1
                    q = j+1
                    r = k+1
                    z[i, j, k] = (x[0]-p)**2 + (x[1]-q)**2 + (x[2]-r)**2 - 0.0625

        return np.min(z)


class G13(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.name = "G13"
        self.num_dim: int = 5
        self.num_obj = 1
        self.num_cstr = 3
        self.num_fidelity = 1

        self.cstr_types = [
            "eq",
            "eq",
            "eq",
        ]

        self.bounds = np.array([
            [-2.3, 2.3],
            [-2.3, 2.3],
            [-3.2, 3.2],
            [-3.2, 3.2],
            [-3.2, 3.2],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.h1,
            self.h2,
            self.h3,
        ]

        # x* = (−1.717143, 1.595709, 1.827247, −0.7636413, −0.763645), f (x*) = 0.0539498


    def objective(self, x):
        return np.exp(np.prod(x))

    def h1(self, x):
        return np.sum(x**2) - 10

    def h2(self, x):
        return x[1]*x[2] - 5*x[3]*x[4]

    def h3(self, x):
        return x[0]**3 + x[1]**3 + 1


class WeldedBeamDesign(BenchmarkProblem):

    def __init__(self):
        super().__init__()
        # TODO: review implementation (different sources = different results)

        self.name = "WeldedBeamDesign"
        self.num_dim: int = 4
        self.num_obj = 1
        self.num_cstr = 6
        self.num_fidelity = 1

        self.bounds = np.array([
            [0.125, 10],
            [0.1, 10],
            [0.1, 10],
            [0.1, 10],
        ])

        self.tags = [
            "GOTestProblems",
        ]

        self.constraints = [
            self.g1,
            self.g2,
            self.g3,
            self.g4,
            self.g5,
            self.g6,
        ]

        self.P = 6_000
        self.L = 14
        self.E = 30e+6
        self.G = 12e+6
        self.t_max = 13_600
        self.s_max = 30_000
        self.d_max = 0.25


    def M(self, x):
        return self.P * (self.L + x[1] / 2)

    def R(self, x):
        return np.sqrt(0.25 * (x[1] ** 2 + (x[0] + x[2]) ** 2))

    def J(self, x):
        # 2 ??? * ??? np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12 + 0.25 * (x[0] + x[2]) ** 2)
        return 2 / np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12 + 0.25 * (x[0] + x[2]) ** 2)

    def P_c(self, x):
        return (4.013 * self.E / (6 * self.L ** 2)) * x[2] * x[3] ** 3 * (1 - 0.25 * x[2] * np.sqrt(self.E / self.G) / self.L)

    def t1(self, x):
        return self.P / (np.sqrt(2) * x[0] * x[1])

    def t2(self, x):
        return self.M(x) * self.R(x) / self.J(x)

    def t(self, x):
        return np.sqrt(self.t1(x) ** 2 + self.t1(x) * self.t2(x) * x[1] / self.R(x) + self.t2(x) ** 2)

    def s(self, x):
        return 6 * self.P * self.L / (x[3] * x[2] ** 2)

    def d(self, x):
        return 4 * self.P * self.L ** 3 / (self.E * x[3] * x[2] ** 3)


    def objective(self, x):
        return 1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14.0 + x[1])

    def g1(self, x):
        return self.t(x) - self.t_max

    def g2(self, x):
        return self.s(x) - self.s_max

    def g3(self, x):
        return x[0] - x[3]

    def g4(self, x):
        return 0.10471*x[0]**2 + 0.04811*x[2]*x[3]*(14.0 + x[1]) - 5.0

    def g5(self, x):
        return self.d(x) - self.d_max

    def g6(self, x):
        return self.P - self.P_c(x)


if __name__ == "__main__":

    from smt_optim.subsolvers.multistart import multistart_minimize

    prob = WeldedBeamDesign()
    # prob.set_dim(20)
    print(prob.bounds)

    sp_cstr = []
    for j in range(prob.num_cstr):
        sp_cstr.append(
            {
                "fun": lambda x, f=prob.constraints[j]: -f(x),
                "type": prob.cstr_types[j] if hasattr(prob, "cstr_types") else "ineq",
            }
        )

    res = multistart_minimize(prob.objective, bounds=prob.bounds, constraints=sp_cstr, n_start=100, seed=0)

    print(res.x)
    print(res.fun)

    # G2 problem
    # x_star = np.array([3.162460616, 3.128331428, 3.094792129, 3.061451495, 3.027929159,2.993826067, 2.958668718, 2.921842273, 0.494825115, 0.488357110,0.482316427, 0.476645751, 0.471295508, 0.466230993, 0.461420050,0.456836648, 0.452458769, 0.448267622, 0.444247010, 0.440382860])
    # print(prob.objective(x_star))
    # print(prob.constraints[0](x_star))
    # print(prob.constraints[1](x_star))

    x_star = np.array([0.18756483308730693, 4.053366828472939, 8.731994883504612, 0.2231022567643955])
    print(prob.objective(x_star))
    print(prob.constraints[0](x_star))
    print(prob.constraints[1](x_star))


