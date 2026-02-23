from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """
    Abstract class for surrogate models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self, xt: list[np.ndarray], yt: list[np.ndarray]) -> None:
        raise Exception("train() method not implemented.")

    @abstractmethod
    def predict_values(self, x_pred: np.ndarray) -> np.ndarray:
        raise Exception("predict_value() method not implemented.")

    @abstractmethod
    def predict_variances(self, x_pred: np.ndarray) -> np.ndarray:
        raise Exception("predict_variance() method not implemented.")