import copy
import time

import numpy as np

from smtoptim.core import OptimizationDataset

# from smtoptim.core import Problem


class OptimizationState:

    def __init__(self, problem):

        self.problem = problem

        # self.num_dim: int = problem.num_dim
        # self.num_obj: int = problem.num_obj
        # self.num_cstr: int = problem.num_cstr
        # self.num_fidelity: int = problem.num_fidelity

        # self.design_space = problem.design_space

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
            scaled_sample.x -= self.problem.obj_configs[0].design_space[:, 0]
            scaled_sample.x /= (self.problem.obj_configs[0].design_space[:, 1] - self.problem.obj_configs[0].design_space[:, 0])

            scaled_sample.obj[:] -= self.qoi_step[lvl][:self.problem.num_obj]
            scaled_sample.obj[:] /= self.qoi_factor[lvl][:self.problem.num_obj]

            scaled_sample.cstr[:] -= self.qoi_step[lvl][self.problem.num_obj:self.problem.num_obj+self.problem.num_cstr]
            scaled_sample.cstr[:] /= self.qoi_factor[lvl][self.problem.num_obj:self.problem.num_obj+self.problem.num_cstr]

            self.scaled_dataset.add(scaled_sample)


    def group_by_fidelity(self) -> tuple[list[np.ndarray], list[np.ndarray]]:

        x = []
        qoi = []

        for lvl in range(self.problem.num_fidelity):

            samples = self.scaled_dataset.get_by_fidelity(lvl)

            x_lvl = np.empty((len(samples), self.problem.num_dim))
            qoi_lvl = np.empty((len(samples), self.problem.num_obj + self.problem.num_cstr))

            for idx, sample in enumerate(samples):
                x_lvl[idx, :] = sample.x
                qoi_lvl[idx, :self.problem.num_obj] = sample.obj
                qoi_lvl[idx, self.problem.num_obj:] = sample.cstr

            x.append(x_lvl)
            qoi.append(qoi_lvl)

        return x, qoi


    def build_models(self):

        x_train, qoi_train = self.group_by_fidelity()

        qoi_models = self.obj_models + self.cstr_models

        t0 = time.perf_counter()

        for qoi_idx in range(self.problem.num_obj+self.problem.num_cstr):

            idx_train = []

            for lvl in range(self.problem.num_fidelity):
                idx_train.append(qoi_train[lvl][:, qoi_idx].reshape(-1, 1))

            qoi_models[qoi_idx].train(x_train, idx_train)

        t1 = time.perf_counter()

        self.iter_log["gp_training_time"] = t1 - t0

    def reset_log(self):
        self.iter_log.clear()


def generate_state(problem) -> OptimizationState:
    return OptimizationState(problem)


