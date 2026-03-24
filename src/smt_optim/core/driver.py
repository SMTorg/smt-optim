import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
import warnings
import time
import json
import csv
import copy

import os

from typing import Callable, Type

import pickle

from smt_optim.core.state import State
from smt_optim.surrogate_models import Surrogate
from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core import Sample, OptimizationDataset, Evaluator
from smt_optim.core import State

from smt_optim.utils.initial_design import generate_initial_design
from smt_optim.utils.stop_criteria import check_stop_criteria
from smt_optim.utils.logger import ConsoleLogger, JsonLogger


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
        return factor * (f(x) - step)

    return wrapped


def wrap_array(
    array: np.ndarray, factor: float | np.ndarray = 1.0, step: float | np.ndarray = 0.0
) -> np.ndarray:
    return factor * (array - step)


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
        warnings.warn(
            f"Infill point was outside of the bounds. L1 correction was applied: (initial = {x}; corrected = {x_corrected})."
        )

    return x_corrected


# def compute_rscv(self, cstr_array: np.ndarray, cstr_config: list[ConstraintConfig], g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:
def compute_rscv(
    cstr_array: np.ndarray, cstr_config: list, g_tol: float = 0.0, h_tol: float = 0.0
) -> np.ndarray:

    scv = np.full_like(cstr_array, 0.0)  # Square Constraint Violation

    for c_id, c_config in enumerate(cstr_config):
        if c_config.type in ["less", "greater"]:
            valid_mask = cstr_array[:, c_id] <= g_tol
            scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id] ** 2

        elif c_config.type == "equal":
            valid_mask = np.abs(cstr_array[:, c_id]) <= h_tol
            scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id] ** 2

        else:
            raise Exception(
                f"{c_config.type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'."
            )

    rscv = np.sqrt(scv.sum(axis=1))

    return rscv


@dataclass
class ObjectiveConfig:
    """
    Configuration of the objective function used in the optimization problem.

    This class stores an objective callable(s) together with surrogate
    modeling information used to approximate the objective during optimization.

    Attributes
    ----------
    objective : list[Callable]
        List of an objective functions. Each callable must accept a decision
        variable vector ``x`` and return a scalar objective value. The functions
        must be ordered in increasing level of fidelity.
    type : {"minimize", "maximize"}, default="minimize"
        Specifies whether the objective should be minimized or maximized.
    surrogate : Surrogate or None, default=None
        Surrogate model used to approximate the objective function.
    surrogate_kwargs : dict or None, default=None
        Optional keyword arguments passed to the surrogate model.
    """

    objective: list[Callable]
    surrogate: type[Surrogate]
    type: str = "minimize"  # problem's type -> "minimize" or "maximize")
    surrogate_kwargs: dict | None = None


@dataclass
class ConstraintConfig:
    """
    Configuration of a constraint function used in the optimization problem.

    This class stores a constraint callable(s) together with surrogate
    modeling information used to approximate the constraint during optimization.

    Attributes
    ----------
    constraint : list[Callable]
        List of constraint functions. Each callable must accept a decision
        variable vector ``x`` and return a scalar constraint value. The functions
        must be ordered in increasing level of fidelity.
    type : {"less", "greater", "equal"}, default="minimize"
        Specifies whether the feasible domain is defined when the constraint value
        is less, greater or equal to the configuration value.
    value: float, default=0
        Specifies the value defining the feasible domain.
    surrogate : Surrogate or None, default=None
        Surrogate model used to approximate the constraint function.
    surrogate_kwargs : dict or None, default=None
        Optional keyword arguments passed to the surrogate model.
    """

    constraint: list[Callable]
    type: str = "less"  # "less"-> g <= 0; "greater" -> g >= 0
    value: float = 0  # g <= value (or g >= value if type is " greater")
    surrogate: Surrogate = None
    surrogate_kwargs: dict | None = None


@dataclass
class DriverConfig:
    """
    Optimization driver configuration

    Attributes
    ----------
    max_iter: int
        Maximum number of iteration
    max_budget: float, default=inf
        Maximum budget before termination of the optimization process
    max_time: float, default=inf
        Maximum time before termination of the optimization process
    nt_init: int
        Number of samples in the initial DoE
    xt_init: list[np.ndarray]
        Initial DoE to use. The Numpy array must be of shape (num_sample, num_dimension).
        By providing an initial DoE, the driver will not generate an initial DoE. Cannot be
        used with `nt_init`
    results_dir: str or None, default=None
        Name of the logging directory
    verbose: bool, default=False
        Print optimization information.
    log_doe: bool, default=False
        Log the value of the function values as soon as they are sampled. The values are
        stored in a .csv file.
    log_stats: bool, default=False
        Log optimization statistics at the end of each iteration. The stats are store in
        a .jsonl file.
    scaling: bool, default=True
        Scale the data. The objective is standardized. The constraints are divided by
        their standard deviation. The design variables are normalized between 0 and 1.
    seed: default=None
        Seed for experiment reproducibility
    """
    ctol: float = 1e-4  # tolerance for all constraints
    max_iter: int | None = None  # max number of BO iterations
    max_budget: float = float("inf")  # max BO budget
    max_time: float = float("inf")  # max BO elapsed time
    nt_init: int | None = None  # number of samples in initial DOE (with LHS)
    xt_init: np.ndarray | None = (
        None  # initial training data [np.ndarray(nt, dim), np.ndarray(nt, dim)]
    )
    results_dir: str | None = "bo_result"  # name for the results directory
    verbose: bool = False  # True/False print each iteration informations
    log_doe: bool = False
    log_stats: bool = False
    callback_func: list[Callable] | Callable | None = (
        None  # additional method to call at the end of each iteration
    )
    scaling: bool = True  # standardize the training data
    seed: None = None


class Driver:
    def __init__(self, problem, config, strategy, strategy_kwargs=dict()):

        self.problem = problem
        self.config = config

        self.state = State(problem)
        self.state.dataset.log_data = True

        self.strategy_kwargs = strategy_kwargs
        self.strategy_kwargs["seed"] = config.seed
        self.strategy = strategy(self.state, **self.strategy_kwargs)

        if self.config.log_doe or self.config.log_stats:
            self.config.results_dir = self.make_res_dir(self.config.results_dir)
        else:
            self.config.results_dir = None

        self.evaluator = Evaluator(problem, config.results_dir)

        # setup loggers
        self.loggers = []
        if self.config.verbose:
            self.loggers.append(ConsoleLogger(self.config))
        if self.config.log_stats:
            self.loggers.append(JsonLogger(self.config))


    def iteration(self, state):
        """
        Perform an optimization iteration. An iteration consists of:
        - scaling all the training data
        - building the surrogate models
        - acquiring points to sample (and their associated fidelity level)
        - sampling the expensive-to-evaluate functions

        Attributes
        ----------
        state: State
            Optimization state on which to perform an iteration.

        Returns
        -------
        State
            Return optimization state on which an iteration was performed.

        """
        state.iter += 1

        # scale data
        state.scale_dataset(self.config.scaling)

        # build models
        state.build_models()

        # get infill
        t0 = time.perf_counter()
        infill = self.strategy.get_infill(state)
        t1 = time.perf_counter()
        state.iter_log["acq_opt_time"] = t1 - t0

        for i in range(len(infill)):
            if infill[i] is not None:
                infill[i] *= (
                    self.problem.design_space[:, 1] - self.problem.design_space[:, 0]
                )
                infill[i] += self.problem.design_space[:, 0]
                state.iter_log["fidelity"] = i + 1

        # evaluate infill points
        self.evaluator.sample_func(infill, state)

        # log iteration data
        self.call_loggers(state)

        return state

    def optimize(self):
        """
        Perform the optimization process which consists of:
        - generating a DoE if the initial dataset is empty
        - while no termination criteria is met, perform iterations


        Returns
        -------
        State
            optimization state on which the optimization process was performed.

        """

        # generate initial design
        if len(self.state.dataset.samples) == 0:
            generate_initial_design(self.state, self.evaluator, self.config)

        # loop - check stop criteria
        while check_stop_criteria(self.state, self.config):
            # iteration
            self.iteration(self.state)

        return self.state

    def finalize(self):
        pass

    def call_loggers(self, state):
        # if self.loggers is not None:
        for logger in self.loggers:
            try:
                logger.on_iter_end(state)
            except Exception as e:
                print(f"Error while logging: {e}")

    def make_res_dir(self, res_dir: str | None) -> str | None:

        og_res_dir = res_dir

        if res_dir is not None:
            # if results_dir already exists, append '_idx' to it to avoid overwriting existing data
            idx = 1
            while os.path.exists(res_dir):
                res_dir = og_res_dir + f"_{idx}"
                idx += 1

            # create results_dir directory
            os.makedirs(res_dir, exist_ok=False)
            return res_dir

        else:
            return None



# # ======= old =======
# class OptimizerOld():
#
#     def __init__(self, obj_config: ObjectiveConfig, config: OptimizerConfig, strategy: AcquisitionStrategy, strategy_params: dict | None = None):
#
#         # initialize print setting
#         self.verbose = config.verbose
#
#         self.opt_data = {}
#
#         self.results_dir = config.results_dir
#         self.log_types = ["doe", "json"]
#
#         # initialize objective function
#         self.obj_config = obj_config
#         self.obj_func = obj_config.objective
#         self.domain = obj_config.design_space
#         self.obj_type = obj_config.type
#         self.obj_models = obj_config.surrogate
#         self.costs = obj_config.costs
#
#         # get constraint configurations
#         self.cstr_config = config.constraints
#         self.cstr_funcs = []
#         self.ctol = config.ctol
#
#         self.iter = 0
#
#         # get stopping criteria
#         self.max_iter = config.max_iter
#         self.max_budget = config.max_budget
#         self.max_time = config.max_time
#
#         # get initial training data / setup
#         self.nt_init = config.nt_init
#         self.xt_init = config.xt_init
#
#         # get misc configuration
#         self.callback_func = config.callback_func
#         self.scaling = config.scaling
#         self.dynamic_costs = config.dynamic_costs
#
#         self._check_optimizer_config()
#         self._setup_logging()
#
#         #
#         self.strategy = strategy
#         self.strategy_params = strategy_params
#
#         self.num_dim = 0
#         self.num_fidelity = 0
#         self.num_obj = 0
#         self.num_cstr = 0
#
#         self.yt_factor = None
#         self.ct_factor = None
#         self.ct_step = None
#
#         self._check_objective()
#         self._check_constraints()
#         self._setup_stopping_criteria()
#         self._check_init_points()
#
#         self.obj_models = []
#         self.cstr_models = []
#         self.g_models = []
#         self.h_models = []
#
#         self._initialize_surrogates()
#
#         self.dataset = OptimizationDataset()
#         self.scaled_dataset = None
#         self.data = []
#         self.xt = []
#         self.yt = []
#         self.ct = []
#         self.f_min = np.inf
#         self.rscv_min = np.inf
#         self.x_min = None
#         self.c_min = None
#         self.samples_time = []
#
#         self.xt_scaled = []
#         self.yt_scaled = []
#         self.ct_scaled = []
#         self.f_min_scaled = np.inf
#
#         # self._check_init_points()
#         # self._gen_init_train_data()
#         # self.update_f_min()
#
#         self.acq_strategy = None
#         self._initialize_acq_strategy()
#
#         self.opt_data = {}
#         self.iter_data = {}
#
#     def _check_optimizer_config(self) -> None:
#
#         if self.dynamic_costs is not None and self.dynamic_costs not in ["samples"]:
#             raise Exception(f"Dynamic costs '{self.dynamic_costs}' is not supported.")
#
#         if callable(self.callback_func):
#             self.callback_func = [self.callback_func]
#
#
#     def _setup_logging(self):
#
#         results_dir = self.results_dir
#
#         if results_dir is not None:
#             # if results_dir already exists, append '_idx' to it to avoid overwriting existing data
#             idx = 1
#             while os.path.exists(results_dir):
#                 results_dir = self.results_dir + f"_{idx}"
#                 idx += 1
#             self.results_dir = results_dir
#
#             # create results_dir directory
#             os.makedirs(self.results_dir)
#
#             # create the DOE subdirectory (used to save the DOEs from each level)
#             doe_path = os.path.join(self.results_dir, "DOE/")
#             os.makedirs(doe_path, exist_ok=True)
#
#
#     def _check_objective(self) -> None:
#
#         if callable(self.obj_func):
#             self.obj_func = [self.obj_func]
#         elif type(self.obj_func) is list:
#             pass
#         else:
#             raise Exception("ObjectiveConfig.objective must be of type list.")
#
#         if self.obj_type == "minimize":
#             maximize = False
#             self.yt_factor = 1.
#         elif self.obj_type == "maximize":
#             maximize = True
#             self.yt_factor = -1.
#         else:
#             raise Exception("ObjectiveConfig.type must be 'minimize' or 'maximize'.")
#
#         # self.obj_func = self._wrap_objectives(self.obj_func, maximize=maximize)
#
#         self.num_dim = self.domain.shape[0]
#         self.num_obj += 1
#
#         self.num_fidelity = len(self.obj_func)
#
#         self.obj_func = [self.obj_func]
#
#         if len(self.costs) != self.num_fidelity:
#             raise Exception("ObjectiveConfig.costs must have the same number of levels as the objective.")
#
#         # TODO: check costs are in ascending order
#
#     def _check_constraints(self) -> None:
#
#         if self.cstr_config is None:
#             self.cstr_config = []
#
#         self.num_cstr = len(self.cstr_config)
#
#         if self.num_cstr > 0:
#
#             self.ct_factor = np.empty(self.num_cstr)
#             self.ct_step = np.empty(self.num_cstr)
#
#             for c_id, c_config in enumerate(self.cstr_config):
#
#                 # self.cstr_funcs.append([])
#
#                 if c_config.type == "greater":
#                     self.ct_factor[c_id] = -1.
#                 else:
#                     self.ct_factor[c_id] = 1.
#
#                 self.ct_step[c_id] = c_config.value
#
#                 if callable(c_config.constraint):
#                     c_config.constraint = [c_config.constraint]
#                 elif type(c_config.constraint) is list:
#                     pass
#                 else:
#                     raise Exception("ConstraintConfig.constraint must be of type list.")
#
#                 if len(c_config.constraint) != self.num_fidelity:
#                     raise Exception("ConstraintConfig.constraint must have the same number of levels as the objective.")
#
#                 self.cstr_funcs.append( c_config.constraint )
#                 #     self._wrap_constraints(c_config.constraint, c_config.type, c_config.value)
#                 # )
#
#     def _setup_stopping_criteria(self):
#
#         if self.num_dim == 0:
#             raise Exception("Problem must have at least one dimension.")
#
#         if self.max_iter is None:
#             warnings.warn("Max number of iterations not specified. Set to 100.")
#             self.max_iter = 10*self.num_dim
#
#         if self.max_budget is None:
#             self.max_budget = np.inf
#
#         if self.max_time is None:
#             self.max_time = np.inf
#
#     def _check_init_points(self):
#
#         if self.nt_init is not None and self.xt_init is not None:
#             raise Exception("Define nt_init or xt_init, but not both.")
#
#         elif self.nt_init is None:
#             self.nt_init = 3*self.num_dim
#
#         if self.xt_init is None:
#             sampler = stats.qmc.LatinHypercube(self.num_dim)
#             xt = sampler.random(self.nt_init)
#             xt = stats.qmc.scale(xt, self.domain[:, 0], self.domain[:, 1])
#             self.xt_init = [xt for _ in range(self.num_fidelity)]
#
#     def _initialize_surrogates(self):
#
#         self.obj_models = [self.obj_config.surrogate(optimizer=self)]
#
#         for c_id, c_config in enumerate(self.cstr_config):
#             self.cstr_models.append(
#                 c_config.surrogate(optimizer=self)
#             )
#
#             if c_config.type in ["less", "greater"]:
#                 self.g_models.append(self.cstr_models[c_id])
#             elif c_config.type == "equal":
#                 self.h_models.append(self.cstr_models[c_id])
#
#     def _gen_init_train_data(self):
#
#         for lvl in range(self.num_fidelity):
#             for idx in range(self.xt_init[lvl].shape[0]):
#                 self.sample_point(self.xt_init[lvl][idx, :], lvl)
#
#     def _initialize_acq_strategy(self):
#
#         acq_context = self.generate_acq_context()
#
#         if type(self.strategy_params) is dict:
#             self.acq_strategy = self.strategy(acq_context, **self.strategy_params)
#         else:
#             self.acq_strategy = self.strategy(acq_context)
#
#         self.acq_strategy.validate_config(acq_context)
#
#     # def update_f_min(self):
#     #     # feasible_mask = np.any(self.ct[-1] <= 1e-4, axis=1)     # use cstr_tol in ConstraintConfig
#     #     # self.f_min = np.min(np.where(feasible_mask == True, self.yt[-1], np.inf))
#     #     # print(f"f_min = {self.f_min}")
#     #
#     #     previous_f_min = self.f_min
#     #
#     #     #feas_mask = np.all(self.ct[-1] <= self.ctol, axis=1)
#     #     valid_cstr = np.full((self.ct[-1].shape[0], self.num_cstr), False)
#     #
#     #     for c_id, c_config in enumerate(self.cstr_config):
#     #         if c_config.type in ["less", "greater"]:
#     #             valid_cstr[:, c_id] = np.where(self.ct[-1][:, c_id] <= self.ctol, True, False)
#     #         elif c_config.type == "equal":
#     #             valid_cstr[:, c_id] = np.where(np.abs(self.ct[-1][:, c_id]) <= self.ctol, True, False)
#     #
#     #     feas_mask = np.all(valid_cstr, axis=1)
#     #
#     #     if np.any(feas_mask):
#     #         local_min_id = self.yt[-1][feas_mask].argmin()
#     #         # global_min_id = feas_mask[local_min_id]
#     #
#     #         self.f_min = self.yt[-1][feas_mask][local_min_id].item()
#     #         self.x_min = self.xt[-1][feas_mask][local_min_id]
#     #         self.c_min = self.ct[-1][feas_mask][local_min_id]
#     #
#     #     else:
#     #         self.f_min = np.inf
#     #         self.x_min = None
#     #         self.c_min = None
#     #
#     #     self._check_f_min_decreasing(self.f_min, previous_f_min)
#     #
#     # def update_rscv_min(self):
#     #     rscv = compute_rscv(self.ct[-1], self.cstr_config, g_tol=0.0, h_tol=0.0)
#     #     self.rscv_min = rscv.min()
#     #
#     #
#     # def _check_f_min_decreasing(self, next_f_min, previous_f_min):
#     #     if previous_f_min < next_f_min:
#     #         warnings.warn("f_min is increasing.")
#
#     def sample_point(self, x_new: np.ndarray, level: int) -> None:
#
#         x_new = check_bounds(x_new, self.domain)
#
#         obj_values = np.empty(self.num_obj)
#         cstr_values = np.empty(self.num_cstr)
#         times = np.empty(self.num_obj+self.num_cstr)
#
#         def sample_func(x_new: np.ndarray, func: Callable) -> tuple[float, float]:
#             t0 = time.perf_counter()
#             value = func(x_new)
#             t1 = time.perf_counter()
#             return value, t1-t0
#
#         for obj_idx in range(self.num_obj):
#             obj_values[obj_idx], times[obj_idx] = sample_func(x_new, self.obj_func[obj_idx][level])
#
#         for cstr_idx in range(self.num_cstr):
#             cstr_values[cstr_idx], times[self.num_obj+cstr_idx] = sample_func(x_new, self.cstr_funcs[cstr_idx][level])
#
#         sample = Sample(
#             x=x_new,
#             fidelity=level,
#             obj=obj_values,
#             cstr=cstr_values,
#             eval_time=times,
#             metadata={"iter": self.iter}
#         )
#
#         self.dataset.add(sample)
#
#         if "doe" in self.log_types:
#             self.save_sample(sample)
#
#     def _standardize_data(self, data: np.ndarray) -> tuple[np.ndarray | float]:
#
#         mean = data.mean()
#         std = data.std()
#         std_data = (data - mean)/std
#
#         return std_data, mean, std
#
#     def scale_data(self):
#
#         num_qoi = self.num_obj + self.num_cstr
#         qoi_factor = np.empty(num_qoi)
#         qoi_step = np.empty(num_qoi)
#
#         for obj_idx in range(self.num_obj):
#             if self.obj_config.type == "minimize":
#                 qoi_factor[obj_idx] = 1
#             elif self.obj_config.type == "maximize":
#                 qoi_factor[obj_idx] = -1
#
#             qoi_step[obj_idx] = 0
#
#         for cstr_idx in range(self.num_cstr):
#             c_config = self.cstr_config[cstr_idx]
#
#             if c_config.type in ["less", "equal"]:
#                 qoi_factor[self.num_obj+cstr_idx] = 1
#             elif c_config.type in ["greater"]:
#                 qoi_factor[self.num_obj+cstr_idx] = -1
#
#             qoi_step[self.num_obj+cstr_idx] = c_config.value
#
#         self.qoi_factor = qoi_factor
#         self.qoi_step = qoi_step
#
#         self.scaled_dataset = OptimizationDataset()
#
#         for sample in self.dataset.samples:
#
#             scaled_sample = copy.deepcopy(sample)
#
#             # should only normalize real variables
#             scaled_sample.x -= self.domain[:, 0]
#             scaled_sample.x /= (self.domain[:, 1] - self.domain[:, 0])
#             scaled_sample.obj[:] *= self.qoi_factor[:self.num_obj]
#             scaled_sample.cstr[:] *= self.qoi_factor[self.num_obj:self.num_obj+self.num_cstr]
#
#             self.scaled_dataset.add(scaled_sample)
#
#
#     def build_models(self):
#
#         def group_by_fidelity() -> tuple[list[np.ndarray], list[np.ndarray]]:
#
#             x = []
#             qoi = []
#
#             for lvl in range(self.num_fidelity):
#
#                 samples = self.scaled_dataset.get_by_fidelity(lvl)
#
#                 x_lvl = np.empty((len(samples), self.num_dim))
#                 qoi_lvl = np.empty((len(samples), self.num_obj + self.num_cstr))
#
#                 for idx, sample in enumerate(samples):
#                     x_lvl[idx, :] = sample.x
#                     qoi_lvl[idx, :self.num_obj] = sample.obj
#                     qoi_lvl[idx, self.num_obj:] = sample.cstr
#
#                 x.append(x_lvl)
#                 qoi.append(qoi_lvl)
#
#             return x, qoi
#
#         x_train, qoi_train = group_by_fidelity()
#
#         qoi_models = self.obj_models + self.cstr_models
#
#         for qoi_idx in range(self.num_obj+self.num_cstr):
#
#             idx_train = []
#
#             for lvl in range(self.num_fidelity):
#                 idx_train.append(qoi_train[lvl][:, qoi_idx].reshape(-1, 1))
#
#             qoi_models[qoi_idx].train(x_train, idx_train)
#
#
#     def generate_acq_context(self):
#
#         cstr_types = []
#         for c_id, c_config in enumerate(self.cstr_config):
#             if c_config.type in ["less", "greater"]:
#                 cstr_types.append("less")
#             elif c_config.type == "equal":
#                 cstr_types.append("equal")
#
#         acq_context = State(
#             num_dim=self.num_dim,
#             num_obj=1,
#             num_cstr=self.num_cstr,
#             num_fidelity=self.num_fidelity,
#             design_space=np.array([[0, 1]] * self.num_dim),
#             obj_models=self.obj_models,
#             cstr_models=self.cstr_models,
#             cstr_types=cstr_types,
#             dataset=self.scaled_dataset,
#         )
#
#         return acq_context
#
#     def sample_infills(self, next_x: list[np.ndarray]) -> None:
#
#         # Convert the single fidelity acquisition function output to a list as
#         # to make it compatible with the multi-fidelity approach
#         if type(next_x) is not list:
#             next_x = [next_x]
#
#         if len(next_x) != self.num_fidelity:
#             warnings.warn("")
#
#         # Sample each fidelity level sequentially
#         for k in range(self.num_fidelity):
#
#             # if None -> the fidelity k is not to be sampled
#             if self.next_x[k] is None:
#                 continue
#
#             # unscale infill point
#             if self.scaling:
#                 self.next_x[k] *= (self.domain[:, 1] - self.domain[:, 0])
#                 self.next_x[k] += self.domain[:, 0]
#
#             # Check if the infill point is already in the training data
#             # TODO: what to do if the next infill location is already in the training data?
#             # if np.any(np.all(self.xt[k] == self.next_x[k], axis=1)):
#             #     warnings.warn("Infill point is already in the training data.")
#             #     continue
#
#             # sample objective function and constraints
#             self.sample_point(self.next_x[k], k)
#
#
#     def perform_iteration(self):
#
#         self.iter_data = dict()  # reset iteration data dictionary
#
#         # ------- Wrap training data -------
#         # self._wrap_training_data()
#
#         # ------- Scale training data -------
#         # self.scale_training_data()
#         self.scale_data()
#
#         # ------- Update cost ratio -------
#         self.update_costs()
#
#         # ------- Surrogate models training -------
#         gp_t0 = time.perf_counter()
#
#         self.build_models()
#
#         gp_t1 = time.perf_counter()
#         gp_time = gp_t1 - gp_t0  # elapsed time for training the models
#
#         # log gp training elapsed time
#         self.iter_data["gp_training_time"] = gp_time
#
#         # # ------- Acquisition function optimization -------
#         acq_context = self.generate_acq_context()
#
#         acq_t0 = time.perf_counter()
#
#         # Find enrichment location
#         self.next_x, acq_data = self.acq_strategy.get_infill(acq_context)
#         self.iter_data["acq_data"] = acq_data
#
#         acq_t1 = time.perf_counter()
#         acq_time = acq_t1 - acq_t0  # elapsed time for finding the next acquisition point
#
#         # log acquisition function maximization time
#         self.iter_data["acq_opt_time"] = acq_time
#
#         # ------- Sample infill location -------
#         self.sample_infills(self.next_x)
#
#         if self.callback_func is not None:
#             for func in self.callback_func:
#                 func(self)
#
#
#     def optimize(self):
#
#         bo_start = time.perf_counter()
#
#         # generate initial doe
#         self._gen_init_train_data()
#
#         # self.scale_training_data()
#         self.scale_data()
#
#         # update f_min
#         # self.update_f_min()
#         # self.update_rscv_min()
#
#         iter_id = 0
#         self.iter = iter_id
#
#         # for l in range(self.num_fidelity):
#         #     self.iter_data[f"n{l + 1}"] = len(self.yt[l])
#         #
#         # self.update_costs()
#         # self.budget = self.compute_used_budget()
#         # self.iter_data["budget"] = self.compute_used_budget()
#         # self.iter_data["f_min"] = self.f_min
#         # self.iter_data["x_min"] = self.x_min
#         # self.iter_data["c_min"] = self.c_min
#
#         self.opt_data[0] = self.iter_data
#         self.save_data()
#
#         self.continue_bo = True
#
#         # if self.verbose: print(
#         #     f"| iter= {iter_id}/{self.max_iter} | budget={self.budget:.3f}/{self.max_budget:.3f} | f_min={self.f_min:.3e} | rscv_min={self.rscv_min:.3e} |"
#         #     )
#
#         while self.continue_bo:
#
#             iter_id += 1
#             self.iter = iter_id
#
#             self.perform_iteration()
#
#
#             # update f_min
#             # self.update_f_min()
#             # self.update_rscv_min()
#
#             # for l in range(self.num_fidelity):
#             #     self.iter_data[f"n{l + 1}"] = len(self.yt[l])
#             #
#             # self.iter_data["f_min"] = self.f_min
#             # self.iter_data["x_min"] = self.x_min
#             # self.iter_data["c_min"] = self.c_min
#
#             self.budget = self.compute_used_budget()
#             self.iter_data["budget"] = self.budget
#
#             self.iter_data["costs"] = self.costs
#
#             # elapsed time since optimization start
#             self.bo_time = time.perf_counter() - bo_start
#             self.iter_data["bo_time"] = self.bo_time
#
#             self.continue_bo = self.check_stop_criteria()
#             self.iter_data["continue"] = self.continue_bo
#
#             # add iteration data to the optimization data dictionary
#             self.opt_data[iter_id] = self.iter_data
#
#             # self.dump_pikle_log()
#             self.save_data()
#
#             # Display the iteration number, best feasible objective and fidelity level sampled
#             # if self.verbose : print(
#             #     f"| iter= {iter_id}/{self.max_iter} | budget={self.budget:.3f}/{self.max_budget:.3f} | f_min={self.f_min:.3e} | rscv_min={self.rscv_min:.3e} | lvl={max_level}/{self.num_fidelity - 1} | gp_time={gp_time:.3f} | acq_time={acq_time:.3f}")
#
#             if self.verbose : print(
#                 f"| iter= {iter_id}/{self.max_iter} | budget={self.budget:.3f}/{self.max_budget:.3f}")
#
#             # ------- End of optimization loop -------
#
#         return self.opt_data
#
#     # def dump_pikle_log(self):
#     #     try:
#     #
#     #         path = os.path.join(self.results_dir, "opt_data.pkl")
#     #
#     #         with open(path, 'wb') as file:
#     #             pickle.dump(self.opt_data, file)
#     #
#     #     except Exception as e:
#     #         # TODO: use warnings.warn
#     #         warnings.warn(f"Error while saving optimization data: {e}")
#
#
#     def dump_json_log(self):
#         try:
#
#             path = os.path.join(self.results_dir, "opt_data.json")
#
#             with open(path, 'w') as file:
#                 safe_data = json_safe(self.opt_data)
#                 json.dump(safe_data, file, indent=2)
#
#         except Exception as e:
#             # TODO: use warnings.warn
#             warnings.warn(f"Error while saving optimization data: {e}")
#
#
#     def save_data(self):
#
#         if "json" in self.log_types:
#             self.dump_json_log()
#
#
#
#
#     def save_sample(self, sample: Sample) -> None:
#
#         try:
#             row = dict()
#
#             row["iter"] = self.iter
#             row["budget"] = self.compute_used_budget() # self.budget
#
#             # save variable
#             for i in range(self.num_dim):
#                 row[f"x{i}"] = sample.x[i]
#
#             # save objectives
#             for i in range(self.num_obj):
#                 row[f"{i}"] = sample.obj[i]
#
#             # save constraints
#             for i in range(self.num_cstr):
#                 row[f"c{i}"] = sample.cstr[i]
#
#             row["time"] = np.sum(sample.eval_time)
#
#             path = os.path.join(self.results_dir, "DOE", f"doe_fidelity_{sample.fidelity}.csv")
#             file_exists = os.path.isfile(path)
#
#             # possibly does not work on Windows -> to be tested
#             with open(path, 'a') as file:
#                 writer = csv.DictWriter(file, fieldnames=row.keys())
#
#                 if not file_exists:
#                     writer.writeheader()
#
#                 writer.writerow(row)
#
#         except Exception as e:
#             print(f"Error while saving the DoE: {e}")
#
#
#     def dump_csv_doe(self, level):
#         pass
#
#         try:
#             row = dict()
#
#             row["iter"] = self.iter
#             row["budget"] = self.budget
#
#             x = self.xt[level][-1, :]
#             print(f"x = {x}")
#             for i in range(len(x)):
#                 row[f"x{i}"] = x[i]
#
#             row["f"] = self.data[level][-1, 0]
#
#             for i in range(self.num_cstr):
#                 row[f"c{i}"] = self.data[level][-1, i+1]
#
#             path = f"DoE/{self.log_filename}_{level}.csv"
#             file_exists = os.path.isfile(path)
#             print(f"file exists = {file_exists}")
#
#             with open(path, 'w') as file:
#                 writer = csv.DictWriter(file, fieldnames=row.keys())
#
#                 if not file_exists:
#                     writer.writeheader()
#
#                 writer.writerow(row)
#
#         except Exception as e:
#             print(f"Error while logging the DoE: {e}")
#
#     def compute_used_budget(self):
#
#         budget = 0
#
#         for k in range(self.num_fidelity):
#             samples = self.dataset.get_by_fidelity(k)
#             budget += self.costs[k] * len(samples)
#
#         return budget
#
#     def check_stop_criteria(self):
#
#         if self.iter >= self.max_iter:
#             return False
#         elif self.budget >= self.max_budget:
#             return False
#         elif self.bo_time >= self.max_time:
#             return False
#         else:
#             return True
#
#     def update_costs(self) -> None:
#         """
#         Update the costs of each level.
#          - If set to None, the costs are never updated.
#          - If set to 'samples', the cost of each level corresponds to it's average time to be sampled.
#
#         :return: None
#         """
#
#         if self.dynamic_costs == "samples":
#             for lvl in range(self.num_fidelity):
#                 # average sampling time per level
#                 self.costs[lvl] = self.samples_time[lvl].sum(axis=1).mean().item()
#
#     # def scale_training_data(self):
#     #
#     #     self.xt_scaled = []
#     #     self.yt_scaled = []
#     #     self.ct_scaled = []
#     #
#     #     for lvl in range(self.num_fidelity):
#     #
#     #         self.xt_scaled.append(self.xt[lvl].copy())
#     #
#     #         # transform the objective into a minimization problem
#     #         self.yt_scaled.append(wrap_array(self.yt[lvl], factor=self.yt_factor))
#     #
#     #         if self.num_cstr >= 1:
#     #             # transform the constraints to define the feasible domain as: g <= 0 and h == 0
#     #             self.ct_scaled.append(wrap_array(self.ct[lvl], factor=self.ct_factor, step=self.ct_step))
#     #
#     #         if self.scaling:
#     #             # scale xt between 0 and 1
#     #             self.xt_scaled[lvl][:] -= self.domain[:, 0]
#     #             self.xt_scaled[lvl][:] /= (self.domain[:, 1] - self.domain[:, 0])
#     #
#     #             # update scaled domain boundaries
#     #             self.domain_scaled = np.empty((self.num_dim, 2))
#     #             self.domain_scaled[:, 0] = 0.
#     #             self.domain_scaled[:, 1] = 1.
#     #
#     #
#     #             # scaled objective to unit std
#     #             yt_scaled, yt_mean, yt_std = self._standardize_data(self.yt[lvl])
#     #             self.yt_scaled[lvl] = yt_scaled
#     #
#     #             # update minimum objective
#     #             self.f_min_scaled = (self.f_min - yt_mean) / yt_std
#     #
#     #             # scaled constraints to unit std
#     #             if self.num_cstr >= 1:
#     #                 for c_id in range(self.ct[lvl].shape[1]):
#     #                     self.ct_scaled[lvl][:, c_id] /= self.ct[lvl][:, c_id].std()
#     #         else:
#     #             self.domain_scaled = self.domain.copy()

if __name__ == "__main__":
    pass
