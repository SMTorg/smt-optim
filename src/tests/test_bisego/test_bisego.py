import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

from smt.surrogate_models import KRG
from smt.sampling_methods import Random, LHS
from smt_optim.acquisition_functions.integrated_variance_reduction import (
    integrated_variance_reduction,
)
from scipy.optimize import minimize

from smt_optim.acquisition_strategies.bisego import BiEGO

import numpy as np

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver
from smt_optim.acquisition_strategies import MFSEGO

from smt_optim.benchmarks.registry import list_problems

import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from tests.test_bisego.naivebisego import NaiveBiEGO, InjectData, SimpleEGO
from smt.sampling_methods import LHS
import smt.design_space as ds
import random

from datetime import datetime
import pickle

L=list_problems(num_obj=[2,2],tags=["zdt"])

surrogate=SmtAutoModel

a=1 # Set to 1 for testing, set to 3 for running Benchmark

# Parameters

n_accuracy=100 # Precision on composite acquisition function
seed=420 # Seed for generating the initial DoE
budget_factor=20 # Budget: budget_factor * dim
init_factor=2 # Initial DoE size: init_factor * dim + 1
min_factor=2 # Initial calls to determine min(f1): min_factor * dim + 1 (same for min(f2))
max_so_iter_factor=2 # Max calls to a single-objective subproblem: max_so_iter_factor * dim + 1
soformulation_naive="Normalized"
soformulation_composite="Product"
multi_start_factor=10 # Number of multistart calls for acquisition function optimization: multi_start_factor * dim
test_number=0

parameters={
    "n_accuracy":n_accuracy,
    "seed":seed,
    "budget_factor":budget_factor,
    "init_factor":init_factor,
    "min_factor":min_factor,
    "max_so_iter_factor":max_so_iter_factor,
    "soformulation_composite":soformulation_composite,
    "soformulation_naive":soformulation_naive,
    "multi_start_factor":multi_start_factor,
    "test_number":test_number,
}

class TestIMSEConvergence(unittest.TestCase):
    def test_imse_space_filling(self):
        """
        Start from a random DoE of 10 points in 3D and use IMSE
        to enrich the design up to 20 points. Average over 3 runs.
        Check the average 1D Space entropy of the points with 10 bins.
        """
        np.random.seed(42)

        # 3D objective function
        def f(x):
            return np.sum((x - 0.5) ** 2, axis=1, keepdims=True)

        n_runs = 3
        n_start = 10
        n_end = 20
        n_iters = n_end - n_start

        xlimits = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        bounds = [(0, 1), (0, 1), (0, 1)]
        bins = np.linspace(0, 1, 11)  # 10 bins for 1D entropy

        all_runs_entropies = np.zeros((n_runs, n_iters + 1))

        # We MUST use LHS for stable integration points for IMSE accuracy
        integration_points = LHS(xlimits=xlimits, criterion="ese", seed=42)(200)

        def get_avg_1d_entropy(points):
            entropies = []
            for dim in range(points.shape[1]):
                counts, _ = np.histogram(points[:, dim], bins=bins)
                pk = (counts + 1e-6) / np.sum(counts + 1e-6)
                # Normalize by the number of bins (10) so the max entropy is exactly 1
                entropies.append(entropy(pk, base=len(pk)))
            return np.mean(entropies)

        for run in range(n_runs):
            # 1. Initialize Random DoE with a deterministic seed for this run
            sampling = Random(xlimits=xlimits, seed=777 + int(run))
            xt = sampling(n_start)
            yt = f(xt)

            all_runs_entropies[run, 0] = get_avg_1d_entropy(xt)

            for i in range(n_iters):
                # Train surrogate
                sm = KRG(print_global=False)
                sm.set_training_values(xt, yt)
                sm.train()

                # Optimize IMSE
                def obj(x):
                    return -integrated_variance_reduction(
                        sm,
                        np.atleast_2d(x),
                        integration_points=integration_points,
                        inv_block=True,
                    )[0, 0]

                best_x = None
                best_val = np.inf

                # Multi-start optimization (5 random starts)
                # Seed depends on run and iteration for deterministic variety
                starts = LHS(xlimits=xlimits, criterion="ese", seed=run * 100 + i)(5)
                for x0 in starts:
                    res = minimize(obj, x0=x0, bounds=bounds, method="L-BFGS-B")
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x

                # Enrich
                xt = np.vstack((xt, best_x))
                yt = np.vstack((yt, f(np.array([best_x]))))

                # Record entropy
                all_runs_entropies[run, i + 1] = get_avg_1d_entropy(xt)

        # Average over runs
        avg_entropies = np.mean(all_runs_entropies, axis=0)

        # Plotting the convergence (optional)
        plot = True

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(
                range(n_start, n_end + 1),
                avg_entropies,
                marker="o",
                linestyle="-",
                color="b",
                label="Avg 1D Entropy (3 runs)",
            )
            plt.xlabel("Number of Points in DoE")
            plt.ylabel("Average 1D Space Entropy (10 bins)")
            plt.title("IMSE 3D Space-Filling Convergence")
            plt.grid(True)
            plt.legend()

            # Save plot
            plot_path = os.path.join(os.path.dirname(__file__), "imse_convergence.png")
            plt.savefig(plot_path)
            plt.close()

        # Assert that the space filling improved on average
        self.assertGreater(avg_entropies[-1], avg_entropies[0])


if __name__ == "__main__":
    unittest.main()
