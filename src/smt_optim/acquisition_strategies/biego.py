import numpy as np
from scipy import stats as stats


from smt_optim.acquisition_functions import log_ei
from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core.state import State


from smt_optim.subsolvers import multistart_minimize, mixvar_multistart_minimize

from smt_optim.acquisition_functions.multi_obj import init_bi_obj_ei_cf
from smt_optim.acquisition_strategies.mfsego import build_scipy_constraints


from smt_optim.utils.multi_obj import get_pareto_mask


def ParetoFront(D, Y):
    # Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    mask = get_pareto_mask(Y)
    indices = np.where(mask)[0]
    sorted_indices = indices[np.argsort(Y[indices, 0])]
    return sorted_indices.tolist()


def PositivePart(x):
    return max(x, 0)


def SingleObjectiveNormalized(y, r, s=None):
    if s is None:
        s = np.ones_like(r)
    else:
        s = np.asarray(s)
    return np.max((y - r) / s, axis=-1)


def SingleObjectiveProduct(y, r):
    # Returns a single objective product formulation of the problem
    pos_part = np.maximum(r - y, 0)
    return -np.prod(pos_part**2, axis=-1)


def Norm(p):
    # Returns the 2-norm of a point
    return np.sqrt(sum(x**2 for x in p))


def Dist(p, q):
    # Returns the distance between two points
    return Norm([p[i] - q[i] for i in range(len(p))])


def DistToNeighbors(p1, p2, p3, w):
    # Returns the sum of squared distances from p2 to its neighbors p1 and p3 on the Pareto front, coefficiented by the Weight.
    return (Dist(p1, p2) ** 2 + Dist(p2, p3) ** 2) / (w + 1)


class BiEGO(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()

        self.acq_func1 = kwargs.get(
            "acq_func", log_ei
        )  # Acquisition function for min(f1) (to be modified to take only f1 as a parameter)
        self.acq_func2 = kwargs.get(
            "acq_func", log_ei
        )  # Acquisition function for min(f2) (same for f2)
        self.acq_func_gen3 = kwargs.get(
            "acq_func_bi", init_bi_obj_ei_cf
        )  # Composite acquisition function for min(f1,f2)
        self.n_multi_start = kwargs.pop("n_multi_start", 5)
        self.n_accuracy = kwargs.pop("n_accuracy", 1000)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))
        self.soformulation = kwargs.pop("so_formulation", "Product")
        self.current_calls = 0
        self.current_subcalls = 0
        self.n_init = kwargs.pop("n_init", self.n_multi_start)
        self.single_obj_max_calls = kwargs.pop("single_obj_max_calls", self.n_init)
        self.min_max_calls = kwargs.pop("min_max_calls", self.n_init)
        self.acq_func_gen1 = lambda state, kwargs: (
            lambda x: self.acq_func1(
                state.obj_models[0].predict_values(x),
                state.obj_models[0].predict_variances(x),
                min(state.scaled_dataset.export_data([0], 0)),
            )[0][0]
        )
        self.acq_func_gen2 = lambda state, kwargs: (
            lambda x: self.acq_func2(
                state.obj_models[1].predict_values(x),
                state.obj_models[1].predict_variances(x),
                min(state.scaled_dataset.export_data([1], 0)),
            )[0][0]
        )

        self.r = None
        self.state = state
        self.X = None
        self.W = None

    def validate_config(self, state):
        pass

    def get_scaled_DoE(self):
        Y = self.state.scaled_dataset.export_as_dict()["obj"]
        D = self.state.scaled_dataset.export_as_dict()["x"]
        return (D, Y)

    def get_DoE(self):
        Y = self.state.dataset.export_as_dict()["obj"]
        D = self.state.dataset.export_as_dict()["x"]
        return (D, Y)

    def get_pareto_front(self):
        D, Y = self.get_scaled_DoE()
        self.X = ParetoFront(D, Y)

    def select_reference_point(self):
        self.get_pareto_front()
        J = len(self.X)
        D, Y = self.get_scaled_DoE()
        X = self.X
        W = self.W
        if J > 2:
            # Select a point of the Pareto front relatively far from its neighbors
            j = max(
                (DistToNeighbors(Y[X[k - 1]], Y[X[k]], Y[X[k + 1]], W[X[k]]), k)
                for k in range(1, J - 1)
            )[1]
            r = (Y[X[j + 1]][0], Y[X[j - 1]][1])
        elif J == 2:
            j = 1
            r = (Y[X[1]][0], Y[X[0]][1])
        elif J == 1:
            return None
        else:
            raise ValueError("The Pareto Front is empty")
        self.W[X[j]] += 1
        return r

    def get_infill(self, state):
        old_pareto_front = self.X
        self.get_pareto_front()

        if self.current_calls == 0:
            self.W = [0 for x in range(len(self.state.dataset.export_as_dict()["x"]))]
            print("Min(f1) phase")

        # Init
        if self.current_calls < self.min_max_calls:
            self.current_calls += 1
            self.W.append(0)
            return self.get_infill_custom(state, self.acq_func_gen1)
        elif self.current_calls < 2 * self.min_max_calls:
            if self.current_calls == self.min_max_calls:
                print("Min(f2) phase")
            self.current_calls += 1
            self.W.append(0)
            return self.get_infill_custom(state, self.acq_func_gen2)

        # Main loop
        else:
            if (
                self.current_calls == 2 * self.min_max_calls
                or self.current_subcalls == 0
                or self.current_subcalls == self.single_obj_max_calls
                or old_pareto_front != self.X
            ):
                print("The Pareto front is of length", len(self.X))
                self.current_subcalls = 0
                r = self.select_reference_point()
                if r is None:
                    self.current_subcalls += 1
                    self.current_calls += 1
                    self.W.append(0)
                    if self.current_calls % 2:
                        return self.get_infill_custom(state, self.acq_func_gen1)
                    else:
                        return self.get_infill_custom(state, self.acq_func_gen2)
                print("Bi-objective phase with r =", r)
                if self.soformulation == "Normalized":
                    self.phi = lambda y: SingleObjectiveNormalized(y, r)
                elif self.soformulation == "Product":
                    self.phi = lambda y: SingleObjectiveProduct(y, r)
                else:
                    raise ValueError("Unknown single-objective formulation")
            self.current_subcalls += 1
            self.current_calls += 1
            self.W.append(0)
            return self.get_infill_custom(
                state, self.acq_func_gen3, phi=self.phi, n_accuracy=self.n_accuracy
            )

    def get_infill_custom(self, state, acq_func_gen, **kwargs):
        self.seed = state.iter

        ac_func = acq_func_gen(state, kwargs)

        def sp_wrapper(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            val = -ac_func(x)
            if isinstance(val, np.ndarray):
                return val.item()
            return val

        scipy_cstr = build_scipy_constraints(state)

        mix_var = False
        for dv in state.problem.design_space.design_variables:
            if dv.__class__.__name__ != "FloatVariable":
                mix_var = True
                break

        if not mix_var:
            sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
            multi_x0 = sampler.random(self.n_multi_start)
            res = multistart_minimize(
                sp_wrapper,
                bounds=np.array([[0, 1]] * state.problem.num_dim),
                constraints=scipy_cstr,
                n_start=self.n_multi_start,
                multi_x0=multi_x0,
                seed=self.seed,
                tol=self.sp_tol,
                method=self.sp_method,
            )
        else:
            res = mixvar_multistart_minimize(
                sp_wrapper,
                design_space=state.problem.design_space,
                constraints=scipy_cstr,
                n_start=self.n_multi_start,
                method=self.sp_method,
                tol=self.sp_tol,
                seed=self.seed,
            )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill
