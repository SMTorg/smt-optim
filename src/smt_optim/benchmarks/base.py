from abc import ABC
from typing import Callable

import numpy as np


class BenchmarkProblem(ABC):
    name: str = None
    num_dim: int | str = None
    num_obj: int = None
    num_cstr: int = None
    num_fidelity: int = None

    bounds: np.ndarray = None
    objective: Callable | list[Callable] = None
    constraints: list = None

    tags: list = None

    def __init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def __repr__(self):
        return f"<{self.name}: num_dim={self.num_dim}, num_cstr={self.num_cstr}, num_fidelity={self.num_fidelity}>"

    def set_dim(self, dim):
        if "n_variable" in self.tags:
            self.num_dim = dim
            self.bounds = self.bounds[-1, :].reshape(1, 2)
            self.bounds = self.bounds.repeat(dim, axis=0)
        else:
            raise Exception("Not a variable dimension problem.")


from pymoo.core.problem import Problem as PymooProblem

class PymooWrapper(PymooProblem):
    """
    Wraps an SMT-Optim BenchmarkProblem into a pymoo Problem so that it can be
    optimized using pymoo algorithms (like NSGA2) to find reference Pareto fronts.
    """
    def __init__(self, benchmark: BenchmarkProblem):
        self.benchmark = benchmark
        
        n_var = benchmark.num_dim
        n_obj = benchmark.num_obj
        
        if hasattr(benchmark, 'constraints') and benchmark.constraints is not None:
            n_ieq_constr = len(benchmark.constraints)
        else:
            n_ieq_constr = 0
            
        xl = benchmark.bounds[:, 0]
        xu = benchmark.bounds[:, 1]
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        num_pt = x.shape[0]
        
        # Evaluate objectives
        out["F"] = np.full((num_pt, self.n_obj), np.nan)
        for i in range(self.n_obj):
            if isinstance(self.benchmark.objective, list):
                obj_func = self.benchmark.objective[i]
            else:
                obj_func = self.benchmark.objective
            # Some benchmark objective functions return 1D arrays for multiple points
            val = obj_func(x)
            out["F"][:, i] = np.atleast_1d(val).ravel()
            
        # Evaluate constraints (assumed to be <= 0 in pymoo)
        if self.n_ieq_constr > 0:
            out["G"] = np.full((num_pt, self.n_ieq_constr), np.nan)
            for i in range(self.n_ieq_constr):
                cstr_func = self.benchmark.constraints[i]
                val = cstr_func(x)
                out["G"][:, i] = np.atleast_1d(val).ravel()
