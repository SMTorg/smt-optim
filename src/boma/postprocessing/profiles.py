import os
import numpy as np
from warnings import warn

""" 
<--- WORK IN PROGRESS --->
"""

def read_instances(algos: list[str], metric: str ="iter", fmin: str ="fmin") -> dict:

    data = {}

    for algo in algos:

        data[algo] = {}

        for root, _, filenames in os.walk(algo):

            for filename in filenames:

                full_path = os.path.join(root, filename)

                with open(full_path, 'r') as file:

                    # get instance name
                    instance_name, ext = os.path.splitext(filename)

                    # get metric and fmin column index from the .csv header
                    headers = file.readline().strip().split(",")
                    metric_index = headers.index(metric)
                    fmin_index = headers.index(fmin)

                    # get instance data
                    inst_data = np.loadtxt(file, delimiter=',', skiprows=0)

                    # extract metric and fmin columns
                    data[algo][instance_name] = np.empty((inst_data.shape[0], 2))
                    data[algo][instance_name][:, 0] = inst_data[:, metric_index]
                    data[algo][instance_name][:, 1] = inst_data[:, fmin_index]

    return data


def get_f0_fstar(data: dict) -> tuple[dict, dict]:

    algos = list(data.keys())
    instances = data[algos[0]].keys()

    f0 = {}
    fstar = {}

    for inst in instances:

        f0[inst] = data[algos[0]][inst][0, 1]
        fstar[inst] = np.inf

        for algo in algos:

            if f0[inst] != data[algo][inst][0, 1]:
                raise Exception(f"instance = {inst}: f0 is not identical for all algorithms.")

            fstar[inst] = min(fstar[inst], data[algo][inst][-1, 1])

    return f0, fstar


def profile(data: dict, tau: float = 0.1, type: str = "perf", dim: dict = None):

    algos = list(data.keys())
    instances = data[algos[0]].keys()

    if dim is None:
        if type == "data":
            warn("Data profiles require the problem's dimensions to be given. n is set to 0 for all instances (n+1 = 1).")
        dim = {}
        for inst in instances:
            dim[inst] = 0

    f0, fstar = get_f0_fstar(data)

    criteria = {}
    min_metric = {}
    for inst in instances:
        criteria[inst] = f0[inst] - (1-tau)*(f0[inst]-fstar[inst])
        min_metric[inst] = np.inf


    perf = {}

    for algo in algos:

        perf[algo] = {}
        perf[algo]["metric"] = []
        perf[algo]["portion"] = []

        portion = 0

        for inst in instances:

            tau_solved = np.where(data[algo][inst][:, 1] <= criteria[inst], 1, 0)

            if np.any(tau_solved == 1):
                index = np.argmax(tau_solved)
                metric = data[algo][inst][index, 0]

                if type == "data":
                    metric /= (dim[inst] + 1)

                min_metric[inst] = min(min_metric[inst], metric)

                perf[algo]["metric"].append( metric )
                portion += 1
                perf[algo]["portion"].append( portion )

            else:
                perf[algo]["metric"].append( np.nan )
                perf[algo]["portion"].append( portion )

        perf[algo]["metric"] = np.array(perf[algo]["metric"])
        perf[algo]["portion"] = np.array(perf[algo]["portion"], dtype=float)

    max_metric = 0

    for algo in algos:
        for i, inst in enumerate(instances):

            if type == "perf":
                perf[algo]["metric"][i] /= min_metric[inst]

        remove = np.isnan(perf[algo]["metric"])
        perf[algo]["metric"] = perf[algo]["metric"][~remove]
        perf[algo]["portion"] = perf[algo]["portion"][~remove]

        perf[algo]["metric"] = np.sort(perf[algo]["metric"])
        max_metric = max(max_metric, perf[algo]["metric"][-1])

        perf[algo]["portion"] /= len(instances)

    max_metric *= 1.1

    if type == "perf":

        one_sum = 0

        for algo in algos:

            perf[algo]["metric"] = np.insert(perf[algo]["metric"], 0, 1)
            perf[algo]["portion"] = np.insert(perf[algo]["portion"], 0, 0)

            one_sum += np.sum(np.where(perf[algo]["metric"][1:] == 1, 1, 0))

        if one_sum < len(instances):
            raise Exception("Total number of instances with ratio 1 is < 1.")

    elif type == "data":
        for algo in algos:
            perf[algo]["metric"] = np.insert(perf[algo]["metric"], 0, 0)
            perf[algo]["portion"] = np.insert(perf[algo]["portion"], 0, 0)

        pass

    for algo in algos:
        perf[algo]["metric"] = np.append(perf[algo]["metric"], max_metric)
        perf[algo]["portion"] = np.append(perf[algo]["portion"], perf[algo]["portion"][-1])


    return perf

def convergence_profile(instances: dict):

    metric = []

    num_instances = len(list(instances.keys()))

    for inst, inst_data in instances.items():
        metric.extend(inst_data[:, 0])

    metric = np.array(metric)
    metric = np.unique(metric)
    metric = np.sort(metric)

    all_fmin = np.empty((num_instances, len(metric)))

    counter = 0
    for inst, inst_data in instances.items():
        e_metric = inst_data[:, 0]
        e_value = inst_data[:, 1]

        indices = np.searchsorted(e_metric, metric, side='right') - 1

        # ensure the indices are within bounds (>= 0)
        indices = np.clip(indices, 0, len(e_metric) - 1)

        all_fmin[counter, :] = e_value[indices]
        counter += 1

    return metric, all_fmin

def accuracy_profile(data: dict):

    algos = list(data.keys())
    instances = data[algos[0]].keys()

    f0, fstar = get_f0_fstar(data)

    r_a = {}

    for algo in algos:

        r_a[algo] = dict()

        for inst in instances:

            last_f = data[algo][inst][-1, 1]

            r_a[algo].append(
                (last_f - f0)/(fstar - f0)
            )

        r_a[algo] = np.array(r_a[algo]).sort()


def accuracy_profile(data: dict):

    algos = list(data.keys())
    instances = list(data[algos[0]].keys())

    f0, fstar = get_f0_fstar(data)

    accuracies = {}
    portions = {}

    max_acc = 0

    for algo in algos:

        accuracies[algo] = []
        portions[algo] = []

        counter = len(instances)

        for i, inst in enumerate(instances):
            last_f = data[algo][inst][-1, 1]
            acc = (last_f - f0[inst])/(fstar[inst] - f0[inst])
            if acc != 1:
                accuracies[algo].append(acc)
                portions[algo].append(counter)
                counter -= 1

        accuracies[algo] = np.array(accuracies[algo], dtype=float)
        portions[algo] = np.array(portions[algo], dtype=float)

        accuracies[algo] = -np.log10(1 - accuracies[algo])
        accuracies[algo] = np.sort(accuracies[algo])
        portions[algo] /= len(instances)

        max_acc = max(max_acc, np.max(accuracies[algo]))

        accuracies[algo] = np.concatenate(([0], accuracies[algo]))
        portions[algo] = np.concatenate(([1], portions[algo]))

    max_acc *= 1.1
    for algo in algos:
        accuracies[algo] = np.concatenate((accuracies[algo], [max_acc]))
        portions[algo] = np.concatenate((portions[algo], [portions[algo][-1]]))

    accuracy_data = {}
    for algo in algos:
        accuracy_data[algo] = {
            "accuracy": accuracies[algo],
            "portion": portions[algo],
        }

    return accuracy_data