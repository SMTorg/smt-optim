import copy
import time

import numpy as np

from smt_optim.core import OptimizationDataset
from smt_optim.core.sample import Sample
from smt_optim.utils.constraints import compute_rscv

# from smt_optim.core import Problem


class State:
    """
    State of the optimization process at a given moment.

    The optimization state holds information about the optimization process at a given moment.

    Parameters
    ----------
    problem: Problem
        The problem to be optimized.

    Attributes
    ----------
    problem: Problem
        The problem to be optimized.
    iter: int
        Current iteration number.
    budget: float
        Current used budget.
    bo_start: float
        Start time of the optimization problem
    bo_time: float
        Elapsed time of the optimization driver.
    obj_models: list[Surrogate]
        List containing the surrogate modeling the objective(s) function(s).
    cstr_models: list[Surrogate]
        List containing the surrogate modeling the constraint(s) function(s).
    dataset: OptimizationDataset
        The dataset containing all samples from the expensive-to-evaluate functions.
    scaled_dataset: OptimizationDataset
        The scaled dataset.
    iter_log: dict
        Dictionary containing logging data.

    Methods
    -------
    scale_dataset(unit_std: bool)
        Scale data in the dataset. The scaled dataset is accessible using the `scaled_dataset`attribute.
    build_models()
        Builds the surrogate models based on the scale dataset.
    get_best_sample()
        Returns the best sample in the dataset.
    """

    def __init__(self, problem):

        self.problem = problem

        self.iter = 0
        self.budget = 0
        self.bo_start = 0
        self.bo_time = 0

        self.obj_models: list = []
        for obj_config in self.problem.obj_configs:
            self.obj_models.append(obj_config.surrogate())

        self.cstr_models: list = []
        for cstr_config in self.problem.cstr_configs:
            self.cstr_models.append(cstr_config.surrogate())


        self.dataset = OptimizationDataset()
        self.scaled_dataset = None

        self.iter_log = dict()


    def scale_dataset(self, unit_std: bool = False):
        """
        Scales the dataset.

        Parameters
        ----------
        unit_std : bool, optional
            If True, normalize by standard deviation.

        Returns
        -------
        None
        """
        num_qoi = self.problem.num_obj + self.problem.num_cstr
        qoi_factor = [np.empty(num_qoi)] * self.problem.num_fidelity
        qoi_step = [np.empty(num_qoi)] * self.problem.num_fidelity

        # initializes scaled constraint bounds
        self.cstr_equal = np.full(self.problem.num_cstr, np.nan)
        self.cstr_lower = np.full(self.problem.num_cstr, np.nan)
        self.cstr_upper = np.full(self.problem.num_cstr, np.nan)

        for lvl in range(self.problem.num_fidelity):

            for obj_idx in range(self.problem.num_obj):

                data = self.dataset.export_data(obj_idx, lvl)

                if unit_std:
                    factor = np.std(data)
                    step = np.mean(data)
                else:
                    factor = 1
                    step = 0

                if self.problem.obj_configs[obj_idx].type == "minimize":
                    qoi_factor[lvl][obj_idx] = factor
                elif self.problem.obj_config[obj_idx].type == "maximize":
                    qoi_factor[lvl][obj_idx] = -factor

                qoi_step[lvl][obj_idx] = step

            for cstr_idx in range(self.problem.num_cstr):

                qoi_idx = self.problem.num_obj+cstr_idx

                c_config = self.problem.cstr_configs[cstr_idx]

                data = self.dataset.export_data(self.problem.num_obj+cstr_idx, lvl)

                if unit_std:
                    factor = np.std(data)
                    step = np.mean(data)
                else:
                    factor = 1
                    step = 0

                qoi_factor[lvl][qoi_idx] = factor
                qoi_step[lvl][qoi_idx] = step

        # scale the constraint bounds
        for c_idx in range(self.problem.num_cstr):

            c_config = self.problem.cstr_configs[c_idx]
            qoi_idx = self.problem.num_obj + c_idx

            if c_config.equal is not None:
                self.cstr_equal[c_idx] = (c_config.equal - qoi_step[-1][qoi_idx]) / qoi_factor[-1][qoi_idx]
            else:
                if c_config.lower is not None:
                    self.cstr_lower[c_idx] = (c_config.lower - qoi_step[-1][qoi_idx]) / qoi_factor[-1][qoi_idx]
                if c_config.upper is not None:
                    self.cstr_upper[c_idx] = (c_config.upper - qoi_step[-1][qoi_idx]) / qoi_factor[-1][qoi_idx]


        self.qoi_factor = qoi_factor
        self.qoi_step = qoi_step

        self.scaled_dataset = OptimizationDataset()

        for sample in self.dataset.samples:

            scaled_sample = copy.deepcopy(sample)

            lvl = scaled_sample.fidelity

            # should only normalize real variables
            scaled_sample.x -= self.problem.design_space[:, 0]
            scaled_sample.x /= (self.problem.design_space[:, 1] - self.problem.design_space[:, 0])

            scaled_sample.obj[:] -= self.qoi_step[lvl][:self.problem.num_obj]
            scaled_sample.obj[:] /= self.qoi_factor[lvl][:self.problem.num_obj]

            scaled_sample.cstr[:] -= self.qoi_step[lvl][self.problem.num_obj:self.problem.num_obj+self.problem.num_cstr]
            scaled_sample.cstr[:] /= self.qoi_factor[lvl][self.problem.num_obj:self.problem.num_obj+self.problem.num_cstr]

            self.scaled_dataset.add(scaled_sample)


    def build_models(self):
        """
        Builds the surrogate models.

        Returns
        -------
        None
        """

        data = self.scaled_dataset.export_as_dict()

        fidelity = data["fidelity"]
        all_xt = data["x"]
        all_yt = data["obj"]
        all_ct = data["cstr"]

        fidelity_masks = []
        xt = []
        yt = []
        ct = []
        for lvl in range(self.problem.num_fidelity):
            fidelity_masks.append((fidelity == lvl).ravel())
            xt.append(all_xt[fidelity_masks[lvl], :])

        t0 = time.perf_counter()

        for idx in range(self.problem.num_obj):
            yt.append(
                [all_yt[fidelity_masks[lvl], idx].reshape(-1, 1) for lvl in range(self.problem.num_fidelity)]
            )
            kwargs = self.problem.obj_configs[idx].surrogate_kwargs if self.problem.obj_configs[idx].surrogate_kwargs is not None else dict()
            self.obj_models[idx].train(xt, yt[idx], **kwargs)

        for idx in range(self.problem.num_cstr):
            ct.append(
                [all_ct[fidelity_masks[lvl], idx].reshape(-1, 1) for lvl in range(self.problem.num_fidelity)]
            )
            kwargs = self.problem.cstr_configs[idx].surrogate_kwargs if self.problem.cstr_configs[idx].surrogate_kwargs is not None else dict()
            self.cstr_models[idx].train(xt, ct[idx], **kwargs)

        t1 = time.perf_counter()

        self.iter_log["gp_training_time"] = t1 - t0

    # def reset_log(self):
    #     self.iter_log.clear()


    def get_best_sample(self, ctol: float = 1e-4, fidelity: int = -1, scaled: bool = False) -> Sample:
        """
        Returns the best sample based on the objective function value.

        Parameters
        ----------
        ctol : float, optional
            Tolerance for constraint violation. Default is 1e-4.
        fidelity : int, optional
            Fidelity level to consider. If -1, uses the highest fidelity. Default is -1.

        Returns
        -------
        sample : Sample
            The best sample based on the objective function value.
        """
        if fidelity == -1:
            fidelity = self.problem.num_fidelity-1

        coeff = 1 if self.problem.obj_configs[0].type == "minimize" else -1

        if scaled:
            dataset = self.scaled_dataset
        else:
            dataset = self.dataset

        data = dataset.export_as_dict()
        fidelity_mask = (data["fidelity"] == fidelity).ravel()
        yt = data["obj"][:, 0]
        rscv = data["rscv"]
        feasible = rscv <= ctol
        if np.any(feasible):
            # mono-objective only
            filtered_yt = np.where(np.logical_and(fidelity_mask, feasible), yt * coeff, np.inf)
            idx = np.argmin(filtered_yt)
        else:
            filtered_rscv = np.where(fidelity_mask, rscv, np.inf)
            idx = np.argmin(filtered_rscv)

        best_sample = self.dataset.samples[idx]

        return best_sample













