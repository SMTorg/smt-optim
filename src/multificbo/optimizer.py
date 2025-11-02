import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
import warnings
import time

from typing import Any, Callable, List, Optional, Union

import pickle

from multificbo.surrogate_models import Surrogate
from multificbo.acquisition_strategies import AcquisitionStrategy

def wrap_objective(obj_func, maximize=False, logger=None):
    """
    Wrap the objective functions to return the opposite in the case of a maximization problem.

    :param obj_func:
    :return:
    """

    wrapped_obj_func = []

    for func in obj_func:

        def wrapped(x, f=func):
            start = time.perf_counter()
            val = f(x)
            end = time.perf_counter()
            elapsed = end - start

            if logger:
                logger.append(elapsed)  # TODO: review how it's stored

            if maximize:
                return -val
            return val

        wrapped_obj_func.append(wrapped)

    return wrapped_obj_func


def wrap_func(func, factor: float = 1, step: float = 0):
    """
    Wrap function to return coeff * (func - step).

    :param x: Function to wrap.
    :type x: callable

    :param bounds: Multiplicative factor.
    :type bounds: float

    :return: Additive step.
    :rtype: float
    """

    def wrapped(x, f=func):
        return factor*(f(x) - step)

    return wrapped







def wrap_constraints(constraints):
    """
    Wrap the constraint functions so to make them have the form c(x) <= 0

    :param constraints:
    :return:
    """

    wrapped_c = []

    for c in constraints:

        if callable(c["func"]):
            c["func"] = [c["func"]]

        n_level = len(c["func"])

        wrapped_ck = []

        for k in range(n_level):

            f = c["func"][k]
            typ = c.get("type", "less")
            val = c.get("value", 0)

            if typ == "less":
                wrapped_ck.append(lambda x, f=f, val=val: f(x) - val)
            elif typ == "greater":
                wrapped_ck.append(lambda x, f=f, val=val: val - f(x))
            else:
                raise ValueError(f"Unknown constraint type: {typ}. Possible types are 'less' and 'greater'.")

        wrapped_c.append(wrapped_ck)

    return wrapped_c


def check_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Apply L1 correction to x point to make sure it's within the problem's bounds.

    :param x: Infill point.
    :type x: np.ndarray

    :param bounds: Problem boundaries.
    :type bounds: np.ndarray

    :return: The bounds corrected infill point.
    :rtype: np.ndarray
    """

    x_corrected = np.where(x < bounds[:, 0], bounds[:, 0], x)
    x_corrected = np.where(x_corrected > bounds[:, 1], bounds[:, 1], x_corrected)

    if np.any(x != x_corrected):
        warnings.warn(f"Infill point was outside of the bounds. L1 correction was applied: (initial = {x}; corrected = {x_corrected}).")

    return x_corrected


@dataclass
class ObjectiveConfig:
    objective: Union[Callable, List[Callable]]
    domain: Union[np.ndarray]                       # problem bounds np.ndarray(dim, 2) lower = bounds[:, 0], upper = bounds[:, 1]
    type: str = "minimize"                          # problem's type -> "minimize" or "maximize")
    surrogate: Surrogate = None
    costs: List = None                              # cost of each fidelity level

@dataclass
class ConstraintConfig:
    constraint: Union[Callable, List[Callable]]
    type: str = "less"                              # "less"-> g <= 0; "greater" -> g >= 0
    value: float = 0                                # g <= value (or g >= value if type is " greater")
    tol: float = 1e-4                               # does not work. use OptimizerConfig.ctol instead
    surrogate: Surrogate = None

@dataclass
class OptimizerConfig:
    constraints: Optional[List[ConstraintConfig]] = None
    ctol: float = 1e-4                                      # tolerance for all constraints
    max_iter: Optional[int] = None                          # max number of BO iterations
    max_budget: Optional[float] = float("inf")              # max BO budget
    max_time: Optional[float] = float("inf")                # max BO elapsed time
    nt_init: Optional[int] = None                           # number of samples in initial DOE (with LHS)
    xt_init: Optional[Any] = None                           # initial training data [np.ndarray(nt, dim), np.ndarray(nt, dim)]
    log_filename: Optional[str] = "log"                     # name for the logfile -> " log_filename" + ".pkl"
    verbose: Optional[bool] = False                         # True/False print each iteration informations
    callback_func: Optional[Callable] = None                # additional method to call at the end of each iteration


class Optimizer():

    def __init__(self, obj_config: ObjectiveConfig, config: OptimizerConfig, strategy: AcquisitionStrategy):

        # initialize print setting
        self.verbose = config.verbose

        self.opt_data = {}

        self.log_filename = config.log_filename

        # initialize objective function
        self.obj_config = obj_config
        self.obj_func = obj_config.objective
        self.domain = obj_config.domain
        self.obj_type = obj_config.type
        self.obj_surrogate = obj_config.surrogate
        self.costs = obj_config.costs

        # get constraint configurations
        self.cstr_config = config.constraints
        self.cstr_funcs = []
        self.ctol = config.ctol

        # get stopping criteria
        self.max_iter = config.max_iter
        self.max_budget = config.max_budget
        self.max_time = config.max_time

        # get initial training data / setup
        self.nt_init = config.nt_init
        self.xt_init = config.xt_init

        # get misc configuration
        self.log_filename = config.log_filename
        self.verbose = config.verbose
        self.callback_func = config.callback_func

        #
        self.strategy = strategy

        self.num_dim = 0
        self.num_levels = 0
        self.num_cstr = 0

        self._check_objective()
        self._check_constraints()
        self._setup_stopping_criteria()
        self._check_init_points()

        self.obj_surrogate = None
        self.cstr_surrogates = []
        self._initialize_surrogates()

        self.xt = []
        self.yt = []
        self.ct = []
        self.f_min = np.inf
        self.x_min = None
        self.c_min = None

        # self._check_init_points()
        # self._gen_init_train_data()
        # self.update_f_min()

        self.acq_strategy = None
        self._initialize_acq_strategy()

        self.opt_data = {}
        self.iter_data = {}

    def _check_objective(self):

        if callable(self.obj_func):
            self.obj_func = [self.obj_func]
        elif type(self.obj_func) is list:
            pass
        else:
            raise Exception("ObjectiveConfig.objective must be of type list.")

        if self.obj_type == "minimize":
            maximize = False
        elif self.obj_type == "maximize":
            maximize = True
        else:
            raise Exception("ObjectiveConfig.type must be 'minimize' or 'maximize'.")

        self.obj_func = self._wrap_objectives(self.obj_func, maximize=maximize)

        self.num_dim = self.domain.shape[0]
        self.num_levels = len(self.obj_func)

        if len(self.costs) != self.num_levels:
            raise Exception("ObjectiveConfig.costs must have the same number of levels as the objective.")

        # TODO: check costs are in ascending order

    def _wrap_objectives(self, obj_func: list, maximize: bool = False) -> list:

        wrapped_obj_func = []

        if maximize:
            factor = -1
        else:
            factor = 1

        for func in obj_func:
            wrapped_obj_func.append(wrap_func(func, factor=factor))

        return wrapped_obj_func

    def _wrap_constraints(self, cstr_func: list, type: str, value: float):
        """
        Wrap all the fidelity levels of a given constraint to match the constraint's type (less or greater;
        by default = "less") and its value (by default = 0).

        :param cstr_func: List of callable where each callable is a fidelity level.
        :type cstr_func: list[callable]

        :param type: Define constraint's type where "less" is for constraints of type g <= 0 and "greater" is for
                     constraints of type g >= 0.
        :type type: str

        :param value: Limit value of the constraint g <= value
        :type value: float

        :return:
        """

        if type == "less":
            factor = 1
        elif type == "greater":
            factor = -1

        wrapped_cstr_func = []

        for func in cstr_func:
            wrapped_cstr_func.append(wrap_func(func, factor=factor, step=value))

        return wrapped_cstr_func


    def _check_constraints(self):

        if self.cstr_config is None:
            self.cstr_config = []

        self.num_cstr = len(self.cstr_config)

        if self.num_cstr > 0:
            for c_id, c_config in enumerate(self.cstr_config):

                # self.cstr_funcs.append([])

                if callable(c_config.constraint):
                    c_config.constraint = [c_config.constraint]
                elif type(c_config.constraint) is list:
                    pass
                else:
                    raise Exception("ConstraintConfig.constraint must be of type list.")

                if len(c_config.constraint) != self.num_levels:
                    raise Exception("ConstraintConfig.constraint must have the same number of levels as the objective.")

                self.cstr_funcs.append(
                    self._wrap_constraints(c_config.constraint, c_config.type, c_config.value)
                )

    def _setup_stopping_criteria(self):

        if self.num_dim == 0:
            raise Exception("Problem must have at least one dimension.")

        if self.max_iter is None:
            warnings.warn("Max number of iterations not specified. Set to 100.")
            self.max_iter = 100

        if self.max_budget is None:
            self.max_budget = np.inf

        if self.max_time is None:
            self.max_time = np.inf

    def _check_init_points(self):

        if self.nt_init is not None and self.xt_init is not None:
            raise Exception("Define nt_init or xt_init, but not both.")

        elif self.nt_init is None:
            self.nt_init = 3*self.num_dim

        if self.xt_init is None:
            sampler = stats.qmc.LatinHypercube(self.num_dim)
            xt = sampler.random(self.nt_init)
            xt = stats.qmc.scale(xt, self.domain[:, 0], self.domain[:, 1])
            self.xt_init = [xt for _ in range(self.num_levels)]

    def _initialize_surrogates(self):

        self.obj_surrogate = self.obj_config.surrogate(optimizer=self)

        for c_config in self.cstr_config:
            self.cstr_surrogates.append(
                c_config.surrogate(optimizer=self)
            )

    def _gen_init_train_data(self):

        for lvl in range(self.num_levels):

            xt = np.empty((self.xt_init[lvl].shape[0], self.num_dim))
            yt = np.empty((self.xt_init[lvl].shape[0], 1))
            ct = np.zeros((self.xt_init[lvl].shape[0], max(1, self.num_cstr)))

            for i in range(xt.shape[0]):
                xt[i, :] = self.xt_init[lvl][i, :]

                obj_value, cstr_values = self.sample_point(xt[i, :], lvl)
                yt[i, :] = obj_value
                ct[i, :] = cstr_values

            self.xt.append(xt)
            self.yt.append(yt)
            self.ct.append(ct)

    def _initialize_acq_strategy(self):
        self.acq_strategy = self.strategy(optimizer=self)

    def update_f_min(self):
        # feasible_mask = np.any(self.ct[-1] <= 1e-4, axis=1)     # use cstr_tol in ConstraintConfig
        # self.f_min = np.min(np.where(feasible_mask == True, self.yt[-1], np.inf))
        # print(f"f_min = {self.f_min}")

        previous_f_min = self.f_min

        feas_mask = np.all(self.ct[-1] <= self.ctol, axis=1)
        if np.any(feas_mask):
            local_min_id = self.yt[-1][feas_mask].argmin()
            # global_min_id = feas_mask[local_min_id]

            self.f_min = self.yt[-1][feas_mask][local_min_id].item()
            self.x_min = self.xt[-1][feas_mask][local_min_id]
            self.c_min = self.ct[-1][feas_mask][local_min_id]

        else:
            self.f_min = np.inf
            self.x_min = None
            self.c_min = None

        self._check_f_min_decreasing(self.f_min, previous_f_min)


    def _check_f_min_decreasing(self, next_f_min, previous_f_min):
        if previous_f_min < next_f_min:
            warnings.warn("f_min is increasing.")

    def sample_point(self, x_new: np.ndarray, level: int) -> tuple:

        x_new = check_bounds(x_new, self.domain)

        obj = self.obj_func[level](x_new)
        obj = obj.reshape(1, 1)

        cstr = np.zeros((1, max(1, self.num_cstr)))

        for c_id, c_func in enumerate(self.cstr_funcs):
            cstr[0, c_id] = c_func[level](x_new)

        return obj, cstr

    def optimize(self):

        bo_start = time.perf_counter()

        # generate initial doe
        self._gen_init_train_data()

        # update f_min
        self.update_f_min()

        iter_id = 0
        self.iter = iter_id

        for l in range(self.num_levels):
            self.iter_data[f"n{l + 1}"] = len(self.yt[l])

        self.iter_data["budget"] = self.compute_used_budget()
        self.iter_data["f_min"] = self.f_min
        self.iter_data["x_min"] = self.x_min
        self.iter_data["c_min"] = self.c_min

        self.opt_data[0] = self.iter_data
        self.dump_pikle_log()

        self.continue_bo = True

        while self.continue_bo:

            iter_id += 1
            self.iter = iter_id

            self.iter_data = dict()  # reset iteration data dictionary

            # if self.verbose: print(f"======= iter: {iter_id}/{self.max_iter} =======")

            # ------- Surrogate models training -------
            gp_t0 = time.perf_counter()

            # train the objective surrogate model
            self.obj_surrogate.train(self.xt, self.yt)

            # if constrained, train the constraint surrogate models

            for c_id, c_config in enumerate(self.cstr_config):
                ct_all_levels = []
                for lvl in range(self.num_levels):
                    ct_all_levels.append( self.ct[lvl][:, c_id].reshape(-1, 1) )

                self.cstr_surrogates[c_id].train(self.xt, ct_all_levels)

            gp_t1 = time.perf_counter()
            gp_time = gp_t1 - gp_t0  # elapsed time for training the models

            # log gp training elapsed time
            self.iter_data["gp_training_time"] = gp_time

            # if self.verbose: print(f"Elapsed time for training GPs: {gp_time:.3f} s")

            # # ------- Acquisition function optimization -------
            acq_t0 = time.perf_counter()

            # Find enrichment location
            self.next_x = self.acq_strategy.execute_infill_strategy(optimizer=self)

            acq_t1 = time.perf_counter()
            acq_time = acq_t1 - acq_t0  # elapsed time for finding the next acquisition point

            # log acquisition function maximization time
            self.iter_data["acq_opt_time"] = acq_time

            # if self.verbose: print(f"Elapsed time for max acq func: {acq_time:.3f} s")

            # ------- Sample infill location -------

            # Convert the single fidelity acquisition function output to a list as
            # to make it compatible with the multi-fidelity approach
            if type(self.next_x) is not list:
                self.next_x = [self.next_x]

            if len(self.next_x) != self.num_levels:
                warnings.warn("")

            infill_values = []
            infill_x = []
            infill_f = []
            infill_c = []

            # Sample each fidelity level sequentially
            max_level = 0
            for k in range(self.num_levels):

                # if None -> the fidelity k is not to be sampled
                if self.next_x[k] is None:
                    continue

                max_level = k

                # Check if the infill point is already in the training data
                # TODO: what to do if the next infill location is already in the training data?
                if np.any(np.all(self.xt[k] == self.next_x[k], axis=1)):
                    warnings.warn("Infill point is already in the training data.")
                    continue

                # sample objective function and constraints
                next_y, next_c = self.sample_point(self.next_x[k], k)

                infill_x.append(self.next_x[k])
                infill_f.append(next_y)
                infill_c.append(next_c)

                # add infill evaluation to the training data
                self.xt[k] = np.vstack((self.xt[k], self.next_x[k]))
                self.yt[k] = np.append(self.yt[k], next_y)
                self.ct[k] = np.vstack((self.ct[k], next_c))

            if self.callback_func is not None:
                self.callback_func(self)

            # log infill point, objective value and constraints values
            self.iter_data["infill_x"] = infill_x
            self.iter_data["infill_f"] = infill_f
            self.iter_data["infill_c"] = infill_c

            self.iter_data["max_f_level"] = max_level

            # update f_min
            self.update_f_min()

            for l in range(self.num_levels):
                self.iter_data[f"n{l + 1}"] = len(self.yt[l])

            self.iter_data["f_min"] = self.f_min
            self.iter_data["x_min"] = self.x_min
            self.iter_data["c_min"] = self.c_min

            self.budget = self.compute_used_budget()
            self.iter_data["budget"] = self.budget

            # elapsed time since optimization start
            self.bo_time = time.perf_counter() - bo_start
            self.iter_data["bo_time"] = self.bo_time

            self.continue_bo = self.check_stop_criteria()
            self.iter_data["continue"] = self.continue_bo

            # add iteration data to the optimization data dictionary
            self.opt_data[iter_id] = self.iter_data

            self.dump_pikle_log()

            # Display the iteration number, best feasible objective and fidelity level sampled
            if self.verbose : print(
                f"| iter= {iter_id}/{self.max_iter} | budget={self.budget}/{self.max_budget} | f_min={self.f_min:.3e} | lvl={max_level}/{self.num_levels - 1} | gp_time={gp_time:.3f} | acq_time={acq_time:.3f}")

            # ------- End of optimization loop -------

        return self.opt_data

    def dump_pikle_log(self):
        try:
            with open(f"{self.log_filename}.pkl", 'wb') as file:
                pickle.dump(self.opt_data, file)
        except Exception as e:
            raise Warning(f"Error while saving optimization data: {e}")

    def compute_used_budget(self):

        budget = 0

        for k in range(self.num_levels):
            budget += self.costs[k] * self.yt[k].shape[0]

        return budget

    def check_stop_criteria(self):

        if self.iter >= self.max_iter:
            return False
        elif self.budget >= self.max_budget:
            return False
        elif self.bo_time >= self.max_time:
            return False
        else:
            return True


if __name__ == '__main__':
    pass