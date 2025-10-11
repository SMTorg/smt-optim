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


@dataclass
class ObjectiveConfig:
    objective: Union[Callable, List[Callable]]
    domain: Union[np.ndarray]
    type: str = "minimize"
    surrogate: Surrogate = None
    costs: List = None

@dataclass
class ConstraintConfig:
    constraint: Union[Callable, List[Callable]]
    type: str = "less"
    tol: float = 1e-4
    surrogate: Surrogate = None

@dataclass
class OptimizerConfig:
    constraints: Optional[List[ConstraintConfig]] = None
    max_iter: Optional[int] = None
    max_budget: Optional[float] = float("inf")
    max_time: Optional[float] = float("inf")
    nt_init: Optional[int] = None
    xt_init: Optional[Any] = None
    log_filename: Optional[str] = "log"
    verbose: Optional[bool] = False


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

        for func in obj_func:

            def wrapped(x, f=func):
                val = f(x)

                if maximize:
                    return -val
                return val

            wrapped_obj_func.append(wrapped)

        return wrapped_obj_func


    def _check_constraints(self):

        if self.cstr_config is None:
            self.cstr_config = []

        self.num_cstr = len(self.cstr_config)

        if self.num_cstr > 0:
            for c_config in self.cstr_config:

                if callable(c_config.constraint):
                    c_config.constraint = [c_config.constraint]
                elif type(c_config.constraint) is list:
                    pass
                else:
                    raise Exception("ConstraintConfig.constraint must be of type list.")

                if len(c_config.constraint) != self.num_levels:
                    raise Exception("ConstraintConfig.constraint must have the same number of levels as the objective.")

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
            raise Exception("Initial training points must be defined either by setting a number of starting point (nt_init) or by providing starting points (xt_init). Not both.")

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

        feas_mask = np.all(self.ct[-1] <= 1e-4, axis=1)
        if np.any(feas_mask):
            next_f_min = np.min(self.yt[-1][feas_mask])
        else:
            next_f_min = np.inf

        self._check_f_min(self.f_min, next_f_min)
        self.f_min = next_f_min

    def _check_f_min(self, previous_f_min, next_f_min):
        if previous_f_min < next_f_min:
            warnings.warn("f_min is increasing.")

    def sample_point(self, x_new: np.ndarray, level: int) -> tuple:

        obj = self.obj_func[level](x_new)
        obj = obj.reshape(1, 1)

        cstr = np.zeros((1, max(1, self.num_cstr)))

        for i, c_config in enumerate(self.cstr_config):
            cstr[i] = c_config.constraint[level](x_new)

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

        self.opt_data[0] = self.iter_data
        self.dump_pikle_log()

        self.continue_bo = True

        while self.continue_bo:

            iter_id += 1
            self.iter = iter_id

            self.iter_data = dict()  # reset iteration data dictionary

            if self.verbose: print(f"======= iter: {iter_id}/{self.max_iter} =======")

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

            if self.verbose: print(f"Elapsed time for training GPs: {gp_time:.3f} s")

            # # ------- Acquisition function optimization -------
            acq_t0 = time.perf_counter()

            # Find enrichment location
            next_x_all_lvl = self.acq_strategy.execute_infill_strategy(optimizer=self)

            acq_t1 = time.perf_counter()
            acq_time = acq_t1 - acq_t0  # elapsed time for finding the next acquisition point

            # log acquisition function maximization time
            self.iter_data["acq_opt_time"] = acq_time

            if self.verbose: print(f"Elapsed time for max acq func: {acq_time:.3f} s")

            # ------- Sample infill location -------

            # Convert the single fidelity acquisition function output to a list as
            # to make it compatible with the multi-fidelity approach
            if type(next_x_all_lvl) is not list:
                next_x_all_lvl = [next_x_all_lvl]

            if len(next_x_all_lvl) != self.num_levels:
                warnings.warn("")

            infill_values = []

            # Sample each fidelity level sequentially
            max_level = 0
            for k in range(self.num_levels):

                next_x = next_x_all_lvl[k]

                # Assumes that all lower fidelity levels must be sampled
                if next_x is None:
                    continue

                if self.verbose: print(f"sampling level {k + 1}")

                max_level = k

                # Check if the infill point is already in the training data
                # TODO: what to do if the next infill location is already in the training data?
                if np.any(np.all(self.xt[k] == next_x, axis=1)):
                    warnings.warn("Infill point is already in the training data.")
                    continue

                next_y, next_c = self.sample_point(next_x, k)
                infill_values.append(next_y)

                # Sample the objective function at the infill location and add them to the training data
                next_y = self.obj_func[k](next_x)
                self.xt[k] = np.vstack((self.xt[k], next_x))
                self.yt[k] = np.append(self.yt[k], next_y)
                self.ct[k] = np.vstack((self.ct[k], next_c))


            self.iter_data["infill_values"] = infill_values
            self.iter_data["max_f_level"] = max_level

            # update f_min
            self.update_f_min()

            for l in range(self.num_levels):
                self.iter_data[f"n{l + 1}"] = len(self.yt[l])

            # iter_data["x_min"] = self.x_min   # TODO: implement self.x_min
            self.iter_data["f_min"] = self.f_min

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
                f"|  iter= {iter_id}  |  f_min= {self.f_min:.3e}  |  mf lvl= {max_level}/{self.num_levels - 1}  |")

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