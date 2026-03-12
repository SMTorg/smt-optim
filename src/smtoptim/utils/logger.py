import os
import json
import csv

import numpy as np

from .json import json_safe


class ConsoleLogger:
    def __init__(self, config):
        self.config = config

    def on_iter_end(self, state) -> None:

        print(f"iter={state.iter}/{self.config.max_iter}  |  budget={state.budget}/{self.config.max_budget}  |  fid={state.iter_log["fidelity"]}/{state.problem.num_fidelity}  |  gp_time={state.iter_log["gp_training_time"]:.1f}  |  acq_time={state.iter_log["acq_opt_time"]:.1f}")


class DoeLogger:
    def __init__(self, config):
        self.config = config
        self.num_saved = 0

    def on_iter_end(self, state) -> None:

        num_samples = len(state.dataset.samples)

        for idx in range(self.num_saved, num_samples):

            sample = state.dataset.samples[idx]

            try:
                row = dict()

                row["iter"] = sample.metadata["iter"]
                row["budget"] = np.nan  # self.compute_used_budget() # self.budget

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

                path = os.path.join(self.config.results_dir, "DOE")
                os.makedirs(path, exist_ok=True)

                path = os.path.join(self.config.results_dir, "DOE", f"doe_fidelity_{sample.fidelity}.csv")
                file_exists = os.path.isfile(path)

                # possibly does not work on Windows -> to be tested
                with open(path, 'a') as file:
                    writer = csv.DictWriter(file, fieldnames=row.keys())

                    if not file_exists:
                        writer.writeheader()

                    writer.writerow(row)

                self.num_saved += 1

            except Exception as e:
                print(f"Error while saving the DoE: {e}")


class JsonLogger:
    def __init__(self, config):
        self.dir = config.results_dir


    def on_iter_end(self, state) -> None:

        path = os.path.join(self.dir, "opt_data.json")

        os.makedirs(self.dir, exist_ok=True)

        if os.path.exists(path):
            with open(path, "r") as file:
                data = json.load(file)
        else:
            data = dict()

        data[state.iter] = state.iter_log

        with open(path, 'w') as file:
            safe_data = json_safe(data)
            json.dump(safe_data, file, indent=4)



