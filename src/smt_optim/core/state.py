import copy
import time

import numpy as np

from smt_optim.core import OptimizationDataset
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
    cstr_types: list[str]
        List containing the constraint types.
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
        self.cstr_types: list[str] = []
        for cstr_config in self.problem.cstr_configs:
            self.cstr_models.append(cstr_config.surrogate())
            self.cstr_types.append(cstr_config.type)


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
                c_config = self.problem.cstr_configs[cstr_idx]

                data = self.dataset.export_data(self.problem.num_obj+cstr_idx, lvl)

                if unit_std:
                    factor = np.std(data)
                else:
                    factor = 1

                if c_config.type in ["less", "equal"]:
                    qoi_factor[lvl][self.problem.num_obj+cstr_idx] = factor
                elif c_config.type in ["greater"]:
                    qoi_factor[lvl][self.problem.num_obj+cstr_idx] = -factor

                qoi_step[lvl][self.problem.num_obj+cstr_idx] = c_config.value


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


    def get_best_sample(self, ctol=1e-4, fidelity=-1):
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

        best_obj = np.inf
        best_sample = None
        coeff = 1 if self.problem.obj_configs[0].type == "minimize" else -1

        samples = self.dataset.get_by_fidelity(fidelity)

        if self.problem.num_cstr == 0:
            for s in samples:
                if s.obj[0] < best_obj * coeff:
                    best_obj = s.obj[0]
                    best_sample = s
        else:
            for s in samples:
                rscv = compute_rscv(s.cstr.reshape(1, -1), self.cstr_types)
                if rscv <= ctol:
                    if s.obj[0] < best_obj * coeff:
                        best_obj = s.obj[0]
                        best_sample = s

            if best_sample is None:
                min_rscv = np.inf

                for s in samples:
                    rscv = compute_rscv(s.cstr.reshape(1, -1), self.cstr_types)
                    if rscv < min_rscv:
                        best_sample = s

        return best_sample

