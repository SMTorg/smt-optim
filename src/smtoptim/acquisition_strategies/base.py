from abc import ABC, abstractmethod

import numpy as np

class AcquisitionStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate_config(self, acq_context) -> None:
        raise Exception("Configuration validation not implemented.")

    @abstractmethod
    def get_infill(self, acq_context) -> tuple[list[np.ndarray], dict]:
        raise Exception("Acquisition Strategy not implemented.")