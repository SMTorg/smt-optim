from abc import ABC
from typing import Callable

import numpy as np

from pymoo.core.problem import Problem as PymooProblem


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


class PymooWrapper(PymooProblem):
    def __init__(self, problem: BenchmarkProblem):

        self.prob = problem

        if self.prob.bounds is None:
            raise ValueError(
                "PymooWrapper requires defined variable bounds and currently supports only continuous optimization problems."
            )

        n_eq_constr = 0
        if hasattr(self.prob, "h_constraints"):
            n_eq_constr = len(self.prob.h_constraints)

        n_ieq_constr = 0
        if hasattr(self.prob, "constraints") and self.prob.constraints is not None:
            n_ieq_constr = len(self.prob.constraints)

        super().__init__(
            n_var=self.prob.num_dim,
            n_obj=self.prob.num_obj,
            n_eq_constr=n_eq_constr,
            n_ieq_constr=n_ieq_constr,
            xl=self.prob.bounds[:, 0],
            xu=problem.bounds[:, 1],
        )

    def _evaluate(self, x, out, *args, **kwargs):

        num_pt = x.shape[0]

        out["F"] = np.full((num_pt, self.n_obj), np.nan)

        if self.n_eq_constr > 0:
            out["H"] = np.empty((num_pt, self.n_eq_constr))

        if self.n_ieq_constr > 0:
            out["G"] = np.empty((num_pt, self.n_ieq_constr))

        for i in range(num_pt):
            if self.prob.num_fidelity > 1:
                for j in range(self.prob.num_obj):
                    out["F"][i, j] = self.prob.objective[j][-1](x[i, :]).item()

                for j in range(self.n_eq_constr):
                    out["H"][i, j] = self.prob.h_constraints[j][-1](x[i, :]).item()

                for j in range(self.n_ieq_constr):
                    out["G"][i, j] = self.prob.constraints[j][-1](x[i, :]).item()
            else:
                for j in range(self.prob.num_obj):
                    out["F"][i, j] = self.prob.objective[j](x[i, :]).item()

                for j in range(self.n_eq_constr):
                    out["H"][i, j] = self.prob.h_constraints[j](x[i, :]).item()

                for j in range(self.n_ieq_constr):
                    out["G"][i, j] = self.prob.constraints[j](x[i, :]).item()
