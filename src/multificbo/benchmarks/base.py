from abc import ABC, abstractmethod
import numpy as np

class BenchmarkProblem(ABC):

    name: str = None
    num_dim: int = None
    num_cstr: int = None
    num_fidelity: int = None

    bounds: np.ndarray = None
    objective: list = None
    constraints: list = None

    def __init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def __repr__(self):
        return f"<{self.name}: num_dim={self.num_dim}, num_cstr={self.num_cstr}, num_fidelity={self.num_fidelity}>"
