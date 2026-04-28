from dataclasses import dataclass, field
from typing import Callable
import time
import warnings
import csv
import os

import numpy as np


from smt_optim.utils.constraints import compute_rscv

@dataclass
class Sample:
    """
    Store sample data.

    Attributes
    ----------
    x : np.ndarray
        Variable
    obj : np.ndarray
        Objective value(s). Array dimension: (num_obj,)
    cstr : np.ndarray
        Contraint value(s). Array dimension: (num_cstr,)
    eval_time : np.ndarray
        Evaluation times of each QoI. Array dimension: (num_obj+num_cstr,)
    metadata : dict
        Dictionary with sample metadata such as iter, budget and fidelity.
    """
    x: np.ndarray                  # (num_dim,)
    fidelity: int

    obj: np.ndarray | None         # (num_obj,)
    cstr: np.ndarray | None        # (num_cstr,)

    eval_time: np.ndarray | None   # (num_obj + num_cstr,)

    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        string = f"======= sample data =======\n"
        string += f"x =             {self.x}\n"
        string += f"obj =           {self.obj}\n"
        string += f"cstr =          {self.cstr}\n"
        string += f"eval_time =     {self.eval_time}\n"
        string+= f"------- meta data -------\n"
        for key, value in self.metadata.items():
            string += f"{key} =     {value}\n"
        string += f"===========================\n"
        return string


class OptimizationDataset:
    """
    Store samples.

    Attributes
    ----------
    samples : list[Sample]
    num_obj: int
        Number of objectives
    num_cstr: int
        Number of constraints
    num_fidelity: int
        Number of fidelity levels
    fidelities: list
        Fidelity levels sorted in increasing order.
    num_samples: dict
        Number of samples for each fidelity levels.
    """
    def __init__(self):
        self.samples: list[Sample] = []

        self.num_obj: int | None = None
        self.num_cstr: int | None = None
        self.num_fidelity: int = 0

        self.fidelities: list = []
        self.num_samples: dict = dict()


    def add(self, sample: Sample):
        """
        Add a new sample to the dataset.

        Parameters
        ----------
        sample : Sample
            The sample to be added. It should contain objective function values (`obj`)
            and/or constraint function values (`cstr`) for each variable in the problem.

        Notes
        -----
        If no samples have been added yet, the number of objectives and constraints are set to
        the lengths of `sample.obj` and `sample.cstr`, respectively. Subsequent samples must
        have the same number of objectives and constraints as the first sample.

        If the fidelity level of the new sample is not already in the dataset, it is added,
        along with a counter for the number of samples at that fidelity.
        """
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


    def get_by_fidelity(self, lvl: int) -> list[Sample]:
        """
        Fetches all the samples corresponding to the specified fidelity level.

        Parameters
        ----------
        lvl : int
            Fidelity level (starting at 0 for the lowest fidelity level) from which to retrieve samples.
        Returns
        -------
        list[Sample]
            A list of samples of the corresponding fidelity level.
        """
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
        """
        Exports the samples data as a dictionary, including fidelity levels, evaluation times, input values,
        objective function values, constraint function values, and RSCV values.

        Returns
        -------
        dict
            A dictionary containing the following keys:
                - "cstr": an array of shape (num_samples, num_cstr) representing the constraint function values for each sample.
                - "eval_time": an array of shape (num_samples,) representing the total evaluation time for each sample.
                - "fidelity": an array of shape (num_samples,) representing the fidelity level of each sample.
                - "obj": an array of shape (num_samples, num_obj) representing the objective function values for each sample.
                - "rscv": an array of shape (num_samples,) representing the Root Square Constraint Violation value for each sample.
                - "x": an array of shape (num_samples, nvar) representing the input values for each sample.
        """
        # TODO: add metadata to exported dict

        num_sample = len(self.samples)
        fidelity = np.empty((num_sample, 1))
        eval_time = np.empty((num_sample, self.num_obj+self.num_cstr))

        nvar = len(self.samples[0].x)
        xt = np.empty((num_sample, nvar))               # inputs
        yt = np.empty((num_sample, self.num_obj))       # objectives
        ct = np.empty((num_sample, self.num_cstr))      # constraints
        rscv = np.empty(num_sample)                     # constraints

        for idx, sample in enumerate(self.samples):
            fidelity[idx, 0] = sample.fidelity
            eval_time[idx, :] = sample.eval_time
            xt[idx, :] = sample.x
            yt[idx, :] = sample.obj
            ct[idx, :] = sample.cstr
            rscv[idx] = sample.metadata["rscv"]

        data = {
            "cstr": ct,
            "eval_time": eval_time,
            "fidelity": fidelity,
            "obj": yt,
            "rscv": rscv,
            "x": xt,
        }

        return data


def sample_func(x_new: np.ndarray, func: Callable) -> tuple[float, float]:
    """
    Evaluates a given function at a specified point and returns the function value and elapsed time.

    Parameters
    ----------
    x_new : np.ndarray
        Point to sample.
    func : Callable
        Function to evaluate (e.g., objective function, constraint function).

    Returns
    -------
    tuple[float, float]
        A tuple containing:
            - The function value at `x_new`.
            - The elapsed time for sampling the function.

    Notes
    -----
    If the function output is not a scalar float or a 1D numpy array, it will be replaced with NaN.
    """
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
    """
    Evaluate the expensive-to-evaluate functions.

    Attributes
    ----------
    problem: Problem
        Optimization problem.
    res_path: str | None
        DOE logging directory path.

    """
    def __init__(self, problem, res_path: str | None = None):
        self.problem = problem
        self.res_path = res_path


    def sample_func(self, infill: list[np.ndarray | None], state) -> None:
        """
        Sample the problem functions at requested query points and add the samples to the optimization state's
        dataset.

        Parameters
        ----------
        infill: list[np.ndarray | None]
            Query points: each numpy array in the list represents a fidelity level and must have shape
            (num_points, num_dim); if a level is set to None, it will be skipped.
        state: State
            Optimization state object.

        Returns
        -------
        None
        """
        for lvl, x_lvl in enumerate(infill):

            if x_lvl is None:
                continue

            else:
                for idx in range(x_lvl.shape[0]):
                    x_new = x_lvl[idx, :]

                    obj_values = np.empty(self.problem.num_obj)
                    cstr_values = np.empty(self.problem.num_cstr)
                    times = np.empty(self.problem.num_obj + self.problem.num_cstr)

                    # samples objectives
                    for obj_idx in range(self.problem.num_obj):
                        obj_values[obj_idx], times[obj_idx] = sample_func(x_new, self.problem.obj_funcs[obj_idx][lvl])

                    # samples constraints
                    for cstr_idx in range(self.problem.num_cstr):
                        cstr_values[cstr_idx], times[self.problem.num_obj + cstr_idx] = sample_func(x_new,
                                                                                            self.problem.cstr_funcs[cstr_idx][lvl])
                    state.budget += state.problem.costs[lvl]

                    sample = Sample(
                        x=x_new,
                        fidelity=lvl,
                        obj=obj_values,
                        cstr=cstr_values,
                        eval_time=times,
                        metadata={
                            "iter": state.iter,
                            "budget": state.budget,
                            "fidelity": lvl,
                            "rscv": compute_rscv(cstr_values.reshape(1, -1), state.problem.cstr_configs).item()
                        }
                    )

                    # adds sample to dataset
                    state.dataset.add(sample)

                    # logs the sample to the DOE file if DOE logging is enabled
                    if self.res_path is not None:
                        self.log_sample(sample)


    def log_sample(self, sample) -> None:
        """
        Append the sample data to the DOE CSV file.

        This method appends new rows to the existing file at the specified path. If the file does not exist, it will be created with a header row.

        Parameters
        ----------
        sample : Sample
            The sample to log.
        Returns
        -------
        None
        """
        try:
            row = dict()

            row["iter"] = sample.metadata.get("iter", np.nan)
            row["budget"] = sample.metadata.get("budget", np.nan)  # self.compute_used_budget() # self.budget
            row["fidelity"] = sample.metadata.get("fidelity", np.nan)  # self.compute_used_budget() # self.budget

            # save variables
            for i in range(len(sample.x)):
                row[f"x{i}"] = sample.x[i]

            # save objectives
            for i in range(len(sample.obj)):
                row[f"f{i}"] = sample.obj[i]

            # save constraints
            for i in range(len(sample.cstr)):
                row[f"c{i}"] = sample.cstr[i]

            row["time"] = np.sum(sample.eval_time)

            path = os.path.join(self.res_path, "doe.csv")
            file_exists = os.path.isfile(path)

            # possibly does not work on Windows OS -> to be tested
            with open(path, 'a') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

        except Exception as e:
            print(f"Error while saving the DoE: {e}")
