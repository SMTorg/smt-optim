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


    def scale_dataset(self):

        num_qoi = self.problem.num_obj + self.problem.num_cstr
        qoi_factor = np.empty(num_qoi)
        qoi_step = np.empty(num_qoi)

        for obj_idx in range(self.problem.num_obj):
            if self.problem.obj_configs[obj_idx].type == "minimize":
                qoi_factor[obj_idx] = 1
            elif self.problem.obj_config[obj_idx].type == "maximize":
                qoi_factor[obj_idx] = -1

            qoi_step[obj_idx] = 0

        for cstr_idx in range(self.problem.num_cstr):
            c_config = self.problem.cstr_configs[cstr_idx]

            if c_config.type in ["less", "equal"]:
                qoi_factor[self.problem.num_obj+cstr_idx] = 1
            elif c_config.type in ["greater"]:
                qoi_factor[self.problem.num_obj+cstr_idx] = -1

            qoi_step[self.problem.num_obj+cstr_idx] = c_config.value

        self.qoi_factor = qoi_factor
        self.qoi_step = qoi_step

        self.scaled_dataset = OptimizationDataset()

        for sample in self.dataset.samples:

            scaled_sample = copy.deepcopy(sample)

            # should only normalize real variables
            scaled_sample.x -= self.problem.obj_configs[0].design_space[:, 0]
            scaled_sample.x /= (self.problem.obj_configs[0].design_space[:, 1] - self.problem.obj_configs[0].design_space[:, 0])
            scaled_sample.obj[:] *= self.qoi_factor[:self.problem.num_obj]
            scaled_sample.cstr[:] *= self.qoi_factor[self.problem.num_obj:self.problem.num_obj+self.problem.num_cstr]

            self.scaled_dataset.add(scaled_sample)


    def build_models(self):

        def group_by_fidelity() -> tuple[list[np.ndarray], list[np.ndarray]]:

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

        x_train, qoi_train = group_by_fidelity()

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


