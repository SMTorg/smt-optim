import unittest

import numpy as np

from smt_optim.acquisition_functions import init_mean


class DummyModel:
    def predict_values(self, x):
        return 2.0 * x


class DummyState:
    def __init__(self):
        self.obj_models = [DummyModel()]


class TestMean(unittest.TestCase):
    def test_init_mean(self):

        x = np.array([[1.0], [2.0]])

        np.testing.assert_array_equal(init_mean(DummyState())(x), 2.0 * x)


if __name__ == "__main__":
    unittest.main()
