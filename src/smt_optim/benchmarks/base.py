from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

class BenchmarkProblem(ABC):

    name: str = None
    num_dim: int | str = None
    num_obj: int = None
    num_cstr: int = None
    num_fidelity: int = None

    bounds: np.ndarray = None
    objective: Callable = None
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
            self.bounds = self.bounds[-1, :].reshape(1, self.num_dim)
            self.bounds = self.bounds.repeat(dim, axis=0)
        else:
            raise Exception("Not a variable dimension problem.")
