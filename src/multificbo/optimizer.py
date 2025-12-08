import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
import warnings
import time
import json
import csv

import os

from typing import Callable

import pickle

from multificbo.surrogate_models import Surrogate
from multificbo.acquisition_strategies import AcquisitionStrategy


def wrap_func(func: Callable, factor: float = 1, step: float = 0) -> Callable:
    """
    Wrap function to return factor * (func - step).

    :param func: Function to wrap.
    :type func: Callable

    :param factor: Multiplicative factor.
    :type factor: float

    :param step: Additive factor.
    :type step: float

    :return: Wrapped function.
    :rtype: Callable
    """

    def wrapped(x, f=func):
        return factor*(f(x) - step)

    return wrapped

def wrap_array(array: np.ndarray, factor: float | np.ndarray = 1., step: float | np.ndarray = 0.) -> np.ndarray:
    return factor*(array - step)


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

# def compute_rscv(self, cstr_array: np.ndarray, cstr_config: list[ConstraintConfig], g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:
def compute_rscv(cstr_array: np.ndarray, cstr_config: list, g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:

    scv = np.full_like(cstr_array, 0.0)     # Square Constraint Violation

    for c_id, c_config in enumerate(cstr_config):

        if c_config.type in ["less", "greater"]:
            valid_mask = cstr_array[:, c_id] <= g_tol
            scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

        elif c_config.type == "equal":
            valid_mask = np.abs(cstr_array[:, c_id]) <= h_tol
            scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

        else:
            raise Exception(f"{c_config.type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'.")

    rscv = np.sqrt(scv.sum(axis=1))

    return rscv


@dataclass
class ObjectiveConfig:
    objective: Callable | list[Callable]
    domain: np.ndarray                              # problem bounds np.ndarray(dim, 2) lower = bounds[:, 0], upper = bounds[:, 1]
    type: str = "minimize"                          # problem's type -> "minimize" or "maximize")
    surrogate: Surrogate = None
    costs: list[float] = None                       # cost of each fidelity level

@dataclass
class ConstraintConfig:
    constraint: Callable | list[Callable]
    type: str = "less"                              # "less"-> g <= 0; "greater" -> g >= 0
    value: float = 0                                # g <= value (or g >= value if type is " greater")
    tol: float = 1e-4                               # does not work. use OptimizerConfig.ctol instead
    surrogate: Surrogate = None

@dataclass
class OptimizerConfig:
    constraints: list[ConstraintConfig] | None = None
    ctol: float = 1e-4                              # tolerance for all constraints
    max_iter: int | None = None                     # max number of BO iterations
    max_budget: float = float("inf")                # max BO budget
    max_time: float = float("inf")                  # max BO elapsed time
    nt_init: int | None = None                      # number of samples in initial DOE (with LHS)
    xt_init: np.ndarray | None = None               # initial training data [np.ndarray(nt, dim), np.ndarray(nt, dim)]
    log_filename: str = "log"                       # --- DEPRECATED --- name for the logfile -> " log_filename" + ".pkl"
    results_dir: str = "bo_results"                 # name for the results directory
    verbose: bool = False                           # True/False print each iteration informations
    callback_func: list[Callable] | Callable | None = None      # additional method to call at the end of each iteration
    scaling: bool = False                           # standardize the training data
    dynamic_costs: str | None = None                # use sampling time to update the costs


class Optimizer():

    def __init__(self, obj_config: ObjectiveConfig, config: OptimizerConfig, strategy: AcquisitionStrategy, strategy_params: dict | None = None):

        # initialize print setting
        self.verbose = config.verbose

        self.opt_data = {}

        self.log_filename = config.log_filename     # DEPRECATED -> use results_dir instead
        self.results_dir = config.results_dir

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

        self.iter = 0

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
        self.scaling = config.scaling
        self.dynamic_costs = config.dynamic_costs

        self._check_optimizer_config()
        self._setup_logging()

        #
        self.strategy = strategy
        self.strategy_params = strategy_params

        self.num_dim = 0
        self.num_levels = 0
        self.num_cstr = 0

        self.yt_factor = None
        self.ct_factor = None
        self.ct_step = None

        self._check_objective()
        self._check_constraints()
        self._setup_stopping_criteria()
        self._check_init_points()

        self.obj_surrogate = None
        self.cstr_surrogates = []
        self.g_surrogates = []          # inequality constraints
        self.h_surrogates = []          # equality constraints

        self._initialize_surrogates()

        self.data = []
        self.xt = []
        self.yt = []
        self.ct = []
        self.f_min = np.inf
        self.rscv_min = np.inf
        self.x_min = None
        self.c_min = None
        self.samples_time = []

        self.xt_scaled = []
        self.yt_scaled = []
        self.ct_scaled = []
        self.f_min_scaled = np.inf

        # self._check_init_points()
        # self._gen_init_train_data()
        # self.update_f_min()

        self.acq_strategy = None
        self._initialize_acq_strategy()

        self.opt_data = {}
        self.iter_data = {}

    def _check_optimizer_config(self) -> None:

        if self.dynamic_costs is not None and self.dynamic_costs not in ["samples"]:
            raise Exception(f"Dynamic costs '{self.dynamic_costs}' is not supported.")

        if callable(self.callback_func):
            self.callback_func = [self.callback_func]


    def _setup_logging(self):

        if self.log_filename != "log":
            warnings.warn("'OptimizerConfig.log_filename' is deprecated. Use 'OptimizerConfig.results_dir' instead.")

        results_dir = self.results_dir

        # if results_dir already exists, append '_idx' to it to avoid overwriting existing data
        idx = 1
        while os.path.exists(results_dir):
            results_dir = self.results_dir + f"_{idx}"
            idx += 1
        self.results_dir = results_dir

        # create results_dir directory
        os.makedirs(self.results_dir)

        # create the DOE subdirectory (used to save the DOEs from each level)
        doe_path = os.path.join(self.results_dir, "DOE/")
        os.makedirs(doe_path, exist_ok=True)


    def _check_objective(self) -> None:

        if callable(self.obj_func):
            self.obj_func = [self.obj_func]
        elif type(self.obj_func) is list:
            pass
        else:
            raise Exception("ObjectiveConfig.objective must be of type list.")

        if self.obj_type == "minimize":
            maximize = False
            self.yt_factor = 1.
        elif self.obj_type == "maximize":
            maximize = True
            self.yt_factor = -1.
        else:
            raise Exception("ObjectiveConfig.type must be 'minimize' or 'maximize'.")

        # self.obj_func = self._wrap_objectives(self.obj_func, maximize=maximize)

        self.num_dim = self.domain.shape[0]

        self.num_levels = len(self.obj_func)

        if len(self.costs) != self.num_levels:
            raise Exception("ObjectiveConfig.costs must have the same number of levels as the objective.")

        # TODO: check costs are in ascending order

    def _check_constraints(self) -> None:

        if self.cstr_config is None:
            self.cstr_config = []

        self.num_cstr = len(self.cstr_config)

        if self.num_cstr > 0:

            self.ct_factor = np.empty(self.num_cstr)
            self.ct_step = np.empty(self.num_cstr)

            for c_id, c_config in enumerate(self.cstr_config):

                # self.cstr_funcs.append([])

                if c_config.type == "greater":
                    self.ct_factor[c_id] = -1.
                else:
                    self.ct_factor[c_id] = 1.

                self.ct_step[c_id] = c_config.value

                if callable(c_config.constraint):
                    c_config.constraint = [c_config.constraint]
                elif type(c_config.constraint) is list:
                    pass
                else:
                    raise Exception("ConstraintConfig.constraint must be of type list.")

                if len(c_config.constraint) != self.num_levels:
                    raise Exception("ConstraintConfig.constraint must have the same number of levels as the objective.")

                self.cstr_funcs.append( c_config.constraint )
                #     self._wrap_constraints(c_config.constraint, c_config.type, c_config.value)
                # )

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

        for c_id, c_config in enumerate(self.cstr_config):
            self.cstr_surrogates.append(
                c_config.surrogate(optimizer=self)
            )

            if c_config.type in ["less", "greater"]:
                self.g_surrogates.append(self.cstr_surrogates[c_id])
            elif c_config.type == "equal":
                self.h_surrogates.append(self.cstr_surrogates[c_id])

    def _gen_init_train_data(self):

        for lvl in range(self.num_levels):

            # use num_dim instead?

            xt = np.empty((self.xt_init[lvl].shape[0], self.num_dim))

            self.data.append(np.empty((self.xt_init[lvl].shape[0], self.num_cstr+1)))
            yt = np.empty((self.xt_init[lvl].shape[0], 1))
            ct = np.zeros((self.xt_init[lvl].shape[0], max(1, self.num_cstr)))
            times = np.empty((self.xt_init[lvl].shape[0], self.num_cstr+1))

            for i in range(xt.shape[0]):
                xt[i, :] = self.xt_init[lvl][i, :]

                obj_value, cstr_values, t = self.sample_point(xt[i, :], lvl)
                self.data[lvl][i, 0] = obj_value
                self.data[lvl][i, 1:] = cstr_values
                yt[i, :] = obj_value
                ct[i, :] = cstr_values
                times[i, :] = t

            self.xt.append(xt)

            self.yt.append(yt)
            self.ct.append(ct)

            self.samples_time.append(times)

    def _initialize_acq_strategy(self):
        if type(self.strategy_params) is dict:
            self.acq_strategy = self.strategy(optimizer=self, **self.strategy_params)
        else:
            self.acq_strategy = self.strategy(optimizer=self)

    def update_f_min(self):
        # feasible_mask = np.any(self.ct[-1] <= 1e-4, axis=1)     # use cstr_tol in ConstraintConfig
        # self.f_min = np.min(np.where(feasible_mask == True, self.yt[-1], np.inf))
        # print(f"f_min = {self.f_min}")

        previous_f_min = self.f_min

        #feas_mask = np.all(self.ct[-1] <= self.ctol, axis=1)
        valid_cstr = np.full((self.ct[-1].shape[0], self.num_cstr), False)

        for c_id, c_config in enumerate(self.cstr_config):
            if c_config.type in ["less", "greater"]:
                valid_cstr[:, c_id] = np.where(self.ct[-1][:, c_id] <= self.ctol, True, False)
            elif c_config.type == "equal":
                valid_cstr[:, c_id] = np.where(np.abs(self.ct[-1][:, c_id]) <= self.ctol, True, False)

        feas_mask = np.all(valid_cstr, axis=1)

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

    def update_rscv_min(self):
        rscv = compute_rscv(self.ct[-1], self.cstr_config, g_tol=0.0, h_tol=0.0)
        self.rscv_min = rscv.min()


    def _check_f_min_decreasing(self, next_f_min, previous_f_min):
        if previous_f_min < next_f_min:
            warnings.warn("f_min is increasing.")

    def sample_point(self, x_new: np.ndarray, level: int) -> tuple[np.ndarray]:

        x_new = check_bounds(x_new, self.domain)

        times = np.empty(1+self.num_cstr)    # [obj_time, cstr0_time, cstr1_time, ...]

        t0 = time.perf_counter()
        obj = self.obj_func[level](x_new)
        t1 = time.perf_counter()
        times[0] = t1-t0

        # TODO: review obj_func output
        if type(obj) is float:
            obj = np.array([[obj]])
        elif type(obj) is np.ndarray:
            obj = obj.reshape(1, 1)

        cstr = np.zeros((1, max(1, self.num_cstr)))

        for c_id, c_func in enumerate(self.cstr_funcs):
            t0 = time.perf_counter()
            cstr[0, c_id] = c_func[level](x_new)
            t1 = time.perf_counter()
            times[c_id+1] = t1-t0

        # TODO: save point to DoE log
        self.add_sample_to_doe_log(level, x_new, obj, cstr, np.sum(times))

        return obj, cstr, times

    def _standardize_data(self, data: np.ndarray) -> tuple[np.ndarray | float]:

        mean = data.mean()
        std = data.std()
        std_data = (data - mean)/std

        return std_data, mean, std

    def optimize(self):

        bo_start = time.perf_counter()

        # generate initial doe
        self._gen_init_train_data()

        self.scale_training_data()

        # update f_min
        self.update_f_min()
        self.update_rscv_min()

        iter_id = 0
        self.iter = iter_id

        for l in range(self.num_levels):
            self.iter_data[f"n{l + 1}"] = len(self.yt[l])

        self.update_costs()
        self.budget = self.compute_used_budget()
        self.iter_data["budget"] = self.compute_used_budget()
        self.iter_data["f_min"] = self.f_min
        self.iter_data["x_min"] = self.x_min
        self.iter_data["c_min"] = self.c_min

        self.opt_data[0] = self.iter_data
        self.dump_pikle_log()

        self.continue_bo = True

        if self.verbose: print(
            f"| iter= {iter_id}/{self.max_iter} | budget={self.budget:.3f}/{self.max_budget:.3f} | f_min={self.f_min:.3e} | rscv_min={self.rscv_min:.3e} |"
            )

        while self.continue_bo:

            iter_id += 1
            self.iter = iter_id

            self.iter_data = dict()  # reset iteration data dictionary

            # ------- Wrap training data -------
            # self._wrap_training_data()

            # ------- Scale training data -------
            self.scale_training_data()

            # ------- Update cost ratio -------
            self.update_costs()

            # ------- Surrogate models training -------
            gp_t0 = time.perf_counter()

            # train the objective surrogate model
            self.obj_surrogate.train(self.xt_scaled, self.yt_scaled)

            # if constrained, train the constraint surrogate models

            for c_id, c_config in enumerate(self.cstr_config):
                ct_all_levels = []
                for lvl in range(self.num_levels):
                    ct_all_levels.append( self.ct_scaled[lvl][:, c_id].reshape(-1, 1) )

                self.cstr_surrogates[c_id].train(self.xt_scaled, ct_all_levels)

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

                # unscale infill point
                if self.scaling:
                    self.next_x[k] *= self.domain[:, 1] - self.domain[:, 0]
                    self.next_x[k] += self.domain[:, 0]

                # Check if the infill point is already in the training data
                # TODO: what to do if the next infill location is already in the training data?
                if np.any(np.all(self.xt[k] == self.next_x[k], axis=1)):
                    warnings.warn("Infill point is already in the training data.")
                    continue

                # sample objective function and constraints
                next_y, next_c, next_time = self.sample_point(self.next_x[k], k)

                infill_x.append(self.next_x[k])
                infill_f.append(next_y)
                infill_c.append(next_c)

                # add infill evaluation to the training data
                qoi = np.empty(self.num_cstr+1)
                qoi[0] = next_y
                qoi[1:] = next_c
                self.xt[k] = np.vstack((self.xt[k], self.next_x[k]))
                self.yt[k] = np.vstack((self.yt[k], next_y))
                self.ct[k] = np.vstack((self.ct[k], next_c))
                self.data[k] = np.vstack((self.data[k], qoi))

                self.samples_time[k] = np.vstack((self.samples_time[k], next_time))

                # self.dump_csv_doe(k)

            if self.callback_func is not None:
                for func in self.callback_func:
                    func(self)

            # log infill point, objective value and constraints values
            self.iter_data["infill_x"] = infill_x
            self.iter_data["infill_f"] = infill_f
            self.iter_data["infill_c"] = infill_c

            self.iter_data["max_f_level"] = max_level

            # self._wrap_training_data()

            # update f_min
            self.update_f_min()
            self.update_rscv_min()

            for l in range(self.num_levels):
                self.iter_data[f"n{l + 1}"] = len(self.yt[l])

            self.iter_data["f_min"] = self.f_min
            self.iter_data["x_min"] = self.x_min
            self.iter_data["c_min"] = self.c_min

            self.budget = self.compute_used_budget()
            self.iter_data["budget"] = self.budget

            for lvl in range(self.num_levels):
                self.iter_data[f"avg_f_time_lvl_{lvl}"] = self.samples_time[lvl].sum(axis=1).mean().item()

            self.iter_data["costs"] = self.costs

            # elapsed time since optimization start
            self.bo_time = time.perf_counter() - bo_start
            self.iter_data["bo_time"] = self.bo_time

            self.continue_bo = self.check_stop_criteria()
            self.iter_data["continue"] = self.continue_bo

            # add iteration data to the optimization data dictionary
            self.opt_data[iter_id] = self.iter_data

            self.dump_pikle_log()
            self.dump_json_log()

            # Display the iteration number, best feasible objective and fidelity level sampled
            if self.verbose : print(
                f"| iter= {iter_id}/{self.max_iter} | budget={self.budget:.3f}/{self.max_budget:.3f} | f_min={self.f_min:.3e} | rscv_min={self.rscv_min:.3e} | lvl={max_level}/{self.num_levels - 1} | gp_time={gp_time:.3f} | acq_time={acq_time:.3f}")

            # ------- End of optimization loop -------

        return self.opt_data

    def dump_pikle_log(self):
        try:

            path = os.path.join(self.results_dir, "opt_data.pkl")

            with open(path, 'wb') as file:
                pickle.dump(self.opt_data, file)

        except Exception as e:
            # TODO: use warnings.warn
            warnings.warn(f"Error while saving optimization data: {e}")


    def dump_json_log(self):
        try:

            path = os.path.join(self.results_dir, "opt_data.json")

            with open(path, 'w') as file:
                safe_data = self._json_safe(self.opt_data)
                json.dump(safe_data, file, indent=2)

        except Exception as e:
            # TODO: use warnings.warn
            warnings.warn(f"Error while saving optimization data: {e}")

    def _json_safe(self, obj):
        try:
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj

            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, dict):
                safe = {}
                for k, v in obj.items():
                    try:
                        safe[str(k)] = self._json_safe(v)
                    except:
                        safe[str(k)] = None
                return safe

            if isinstance(obj, (list, tuple)):
                out = []
                for v in obj:
                    try:
                        out.append(self._json_safe(v))
                    except Exception:
                        out.append(None)
                return out

            json.dumps(obj)
            return obj

        except Exception as e:
            warnings.warn(f"Failed to convert: {obj}. Error message: {e}")
            return None


    def add_sample_to_doe_log(self, level: int, x: np.ndarray, obj: float, cstrs: np.ndarray, time: float) -> None:

        try:
            row = dict()

            row["iter"] = self.iter
            row["budget"] = np.nan # self.budget

            for i in range(x.shape[0]):
                row[f"x{i}"] = x[i]

            row["f"] = obj.item()

            for i in range(cstrs.shape[1]):
                row[f"c{i}"] = cstrs[0, i]

            row["time"] = time

            path = os.path.join(self.results_dir, "DOE", f"doe_level_{level}.csv")
            file_exists = os.path.isfile(path)

            # possibly does not work on Windows -> to be tested
            with open(path, 'a') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

        except Exception as e:
            print(f"Error while logging the DoE: {e}")


    def dump_csv_doe(self, level):
        pass

        try:
            row = dict()

            row["iter"] = self.iter
            row["budget"] = self.budget

            x = self.xt[level][-1, :]
            print(f"x = {x}")
            for i in range(len(x)):
                row[f"x{i}"] = x[i]

            row["f"] = self.data[level][-1, 0]

            for i in range(self.num_cstr):
                row[f"c{i}"] = self.data[level][-1, i+1]

            path = f"DoE/{self.log_filename}_{level}.csv"
            file_exists = os.path.isfile(path)
            print(f"file exists = {file_exists}")

            with open(path, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

        except Exception as e:
            print(f"Error while logging the DoE: {e}")

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

    def update_costs(self) -> None:
        """
        Update the costs of each level.
         - If set to None, the costs are never updated.
         - If set to 'samples', the cost of each level corresponds to it's average time to be sampled.

        :return: None
        """

        if self.dynamic_costs == "samples":
            for lvl in range(self.num_levels):
                # average sampling time per level
                self.costs[lvl] = self.samples_time[lvl].sum(axis=1).mean().item()

    def scale_training_data(self):

        self.xt_scaled = []
        self.yt_scaled = []
        self.ct_scaled = []

        for lvl in range(self.num_levels):

            self.xt_scaled.append(self.xt[lvl].copy())

            # transform the objective into a minimization problem
            self.yt_scaled.append(wrap_array(self.yt[lvl], factor=self.yt_factor))

            # transform the constraints to define the feasible domain as: g <= 0 and h == 0
            self.ct_scaled.append(wrap_array(self.ct[lvl], factor=self.ct_factor, step=self.ct_step))

            if self.scaling:
                # scale xt between 0 and 1
                self.xt_scaled[lvl][:] -= self.domain[:, 0]
                self.xt_scaled[lvl][:] /= (self.domain[:, 1] - self.domain[:, 0])

                # update scaled domain boundaries
                self.domain_scaled = np.empty((self.num_dim, 2))
                self.domain_scaled[:, 0] = 0.
                self.domain_scaled[:, 1] = 1.


                # scaled objective to unit std
                yt_scaled, yt_mean, yt_std = self._standardize_data(self.yt[lvl])
                self.yt_scaled[lvl] = yt_scaled

                # update minimum objective
                self.f_min_scaled = (self.f_min - yt_mean) / yt_std

                # scaled constraints to unit std
                for c_id in range(self.ct[lvl].shape[1]):
                    self.ct_scaled[lvl][:, c_id] /= self.ct[lvl][:, c_id].std()
            else:
                self.domain_scaled = self.domain.copy()

if __name__ == '__main__':
    pass