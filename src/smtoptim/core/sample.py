from dataclasses import dataclass, field
from typing import Callable
import time
import warnings

import numpy as np


@dataclass
class Sample:
    x: np.ndarray                  # (num_dim,)
    fidelity: int

    obj: np.ndarray | None         # (num_obj,)
    cstr: np.ndarray | None        # (num_cstr,)

    eval_time: np.ndarray | None   # (num_obj + num_cstr,)

    metadata: dict = field(default_factory=dict)


class OptimizationDataset:
    def __init__(self):
        self.samples: list[Sample] = []

        self.num_obj: int | None = None
        self.num_cstr: int | None = None
        self.num_fidelity: int = 0

        self.fidelities: list = []
        self.num_samples: dict = dict()


    def add(self, sample: Sample):
        self.samples.append(sample)

        if self.num_obj is None:
            self.num_obj = len(sample.obj)
            self.num_cstr = len(sample.cstr) if sample.cstr is not None else 0
        else:
            if len(sample.obj) != self.num_obj or len(sample.cstr) != self.num_cstr:
                raise Exception("Sample data does not match dataset.")

        if sample.fidelity not in self.fidelities:
            self.fidelities.append(sample.fidelity)
            self.num_samples[sample.fidelity] = 0
            self.num_fidelity += 1

        self.num_samples[sample.fidelity] += 1


    def get_by_fidelity(self, lvl: int):
        return [s for s in self.samples if s.fidelity == lvl]


    def export_data(self, idx: int | list[int], lvl: int) -> np.ndarray:

        if isinstance(idx, int):
            idx = [idx]

        data = []

        samples = self.get_by_fidelity(lvl)

        for s in samples:

            row = []

            for i, qoi_idx in enumerate(idx):
                if qoi_idx < self.num_obj:
                    row.append(s.obj[qoi_idx])
                else:
                    row.append(s.cstr[qoi_idx-self.num_obj])

            data.append(row)

        return np.array(data)


    def export_as_dict(self) -> dict:

        num_sample = len(self.samples)
        fidelity = np.empty((num_sample, 1))
        eval_time = np.empty((num_sample, self.num_obj+self.num_cstr))

        nvar = len(self.samples[0].x)
        xt = np.empty((num_sample, nvar))             # inputs
        yt = np.empty((num_sample, self.num_obj))     # objectives
        ct = np.empty((num_sample, self.num_cstr))    # constraints

        for idx, sample in enumerate(self.samples):
            fidelity[idx, 0] = sample.fidelity
            eval_time[idx, :] = sample.eval_time
            xt[idx, :] = sample.x
            yt[idx, :] = sample.obj
            ct[idx, :] = sample.cstr

        data = {
            "fidelity": fidelity,
            "eval_time": eval_time,
            "x": xt,
            "obj": yt,
            "cstr": ct,
        }

        return data


def sample_func(x_new: np.ndarray, func: Callable) -> tuple[float, float]:

    t0 = time.perf_counter()

    output = func(x_new)

    t1 = time.perf_counter()
    elapsed_time = t1 - t0

    if isinstance(output, float):
        pass
    elif isinstance(output, np.ndarray):
        output = output.copy().ravel()
        if len(output) == 1:
            output = output.item()
        else:
            warnings.warn(f"Invalid function output: {output}")
            output = np.nan

    return output, elapsed_time



class Evaluator:
    def __init__(self, problem):
        self.problem = problem

    def sample(self, infill: list[np.ndarray | None], state):

        for lvl, x_lvl in enumerate(infill):

            if x_lvl is None:
                continue

            else:
                for idx in range(x_lvl.shape[0]):
                    x_new = x_lvl[idx, :]

                    obj_values = np.empty(self.problem.num_obj)
                    cstr_values = np.empty(self.problem.num_cstr)
                    times = np.empty(self.problem.num_obj + self.problem.num_cstr)

                    for obj_idx in range(self.problem.num_obj):
                        obj_values[obj_idx], times[obj_idx] = sample_func(x_new, self.problem.obj_funcs[obj_idx][lvl])

                    for cstr_idx in range(self.problem.num_cstr):
                        cstr_values[cstr_idx], times[self.problem.num_obj + cstr_idx] = sample_func(x_new,
                                                                                            self.problem.cstr_funcs[cstr_idx][lvl])

                    sample = Sample(
                        x=x_new,
                        fidelity=lvl,
                        obj=obj_values,
                        cstr=cstr_values,
                        eval_time=times,
                        metadata={"iter": state.iter}
                    )

                    state.dataset.add(sample)