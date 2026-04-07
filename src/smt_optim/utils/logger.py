import os
import json
import csv

import numpy as np

from .json import json_safe


def format_value(v, fmt):
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


class ConsoleLogger:
    def __init__(self, config):
        self.config = config

        self.headers = ["iter", "budget", "fmin", "rscv", "fidelity", "gp_time", "acq_time"]
        width = 14
        self.widths = [max(len(h), width) for h in self.headers]

        self.header_fmt = " ".join(f"{{:>{width}}}" for _ in self.headers)
        self.row_fmt = " ".join(f"{{:>{width}}}" for _ in self.headers)

        self.formats = {
            "iter": ".0f",
            "budget": ".3f",
            "fmin": ".5e",
            "rscv": ".3e",
            "fidelity": ".0f",
            "gp_time": ".3f",
            "acq_time": ".3f",
        }

        self.iter = 0
        self.repeat_header = 10


    def on_iter_end(self, state) -> None:

        if self.iter % self.repeat_header == 0:
            self.print_header()

        sample = state.get_best_sample(ctol=1e-4)

        data = {
            "iter": state.iter,
            "budget": state.budget,
            "fmin": sample.obj[0],
            "rscv": sample.metadata["rscv"],
            "fidelity": state.iter_log["fidelity"],
            "gp_time": state.iter_log["gp_training_time"],
            "acq_time": state.iter_log["acq_opt_time"],
        }
        row = [format_value(data[h], self.formats[h]) for h in self.headers]
        print(self.row_fmt.format(*row))

        self.iter += 1
        #print(f"iter={state.iter}/{self.config.max_iter}  |  budget={state.budget:.2f}/{self.config.max_budget:.2f}  |  fmin={sample.obj[0]:.2e}  |  fid={state.iter_log["fidelity"]}/{state.problem.num_fidelity}  |  gp_time={state.iter_log["gp_training_time"]:.1f}  |  acq_time={state.iter_log["acq_opt_time"]:.1f}")

    def print_header(self):
        print(self.header_fmt.format(*self.headers))


class JsonLogger:
    def __init__(self, config):
        self.dir = config.results_dir


    def on_iter_end(self, state) -> None:

        path = os.path.join(self.dir, "stats.jsonl")

        os.makedirs(self.dir, exist_ok=True)

        with open(path, 'a') as file:
            safe_iter_log = json_safe(state.iter_log)
            file.write(json.dumps(safe_iter_log) + "\n")
            # json.dump(safe_iter_log, file, indent=4)


# class DoeLogger:
#     def __init__(self, config):
#         self.config = config
#         self.num_saved = 0
#
#
#     def log_sample(self, state, sample) -> None:
#         """
#         Log sample data once sampled
#
#         :param state:
#         :param sample:
#         :return:
#         """
#
#         if self.config.results_dir is None:
#             return None
#
#         try:
#             row = dict()
#
#             row["iter"] = sample.metadata["iter"]
#             row["budget"] = np.nan  # self.compute_used_budget() # self.budget
#
#             # save variables
#             for i in range(len(sample.x)):
#                 row[f"x{i}"] = sample.x[i]
#
#             # save objectives
#             for i in range(len(sample.obj)):
#                 row[f"f{i}"] = sample.obj[i]
#
#             # save constraints
#             for i in range(len(sample.cstr)):
#                 row[f"c{i}"] = sample.cstr[i]
#
#             row["time"] = np.sum(sample.eval_time)
#
#             path = os.path.join(self.config.results_dir, "doe.csv")
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
#             self.num_saved += 1
#
#         except Exception as e:
#             print(f"Error while saving the DoE: {e}")
#
#         pass
#
#     def on_iter_end(self, state) -> None:
#         """
#         DOE data should be logged right after sampling the blackbox function (to avoid loss of data)
#
#         :param state:
#         :return:
#         """
#         num_samples = len(state.dataset.samples)
#
#         for idx in range(self.num_saved, num_samples):
#
#             sample = state.dataset.samples[idx]
#
#             try:
#                 row = dict()
#
#                 row["iter"] = sample.metadata["iter"]
#                 row["budget"] = np.nan  # self.compute_used_budget() # self.budget
#
#                 # save variables
#                 for i in range(len(sample.x)):
#                     row[f"x{i}"] = sample.x[i]
#
#                 # save objectives
#                 for i in range(len(sample.obj)):
#                     row[f"f{i}"] = sample.obj[i]
#
#                 # save constraints
#                 for i in range(len(sample.cstr)):
#                     row[f"c{i}"] = sample.cstr[i]
#
#                 row["time"] = np.sum(sample.eval_time)
#
#                 path = os.path.join(self.config.results_dir, "DOE")
#                 os.makedirs(path, exist_ok=True)
#
#                 path = os.path.join(self.config.results_dir, "DOE", f"doe_fidelity_{sample.fidelity}.csv")
#                 file_exists = os.path.isfile(path)
#
#                 # possibly does not work on Windows -> to be tested
#                 with open(path, 'a') as file:
#                     writer = csv.DictWriter(file, fieldnames=row.keys())
#
#                     if not file_exists:
#                         writer.writeheader()
#
#                     writer.writerow(row)
#
#                 self.num_saved += 1
#
#             except Exception as e:
#                 print(f"Error while saving the DoE: {e}")

