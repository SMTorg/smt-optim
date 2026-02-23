from abc import ABC, abstractmethod

import numpy as np

class AcquisitionStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate_config(self, state) -> None:
        raise Exception("Configuration validation not implemented.")

    @abstractmethod
    def get_infill(self, state) -> list[np.ndarray]:
        raise Exception("Acquisition Strategy not implemented.")