import unittest
import numpy as np

from smt_optim.core.state import State
from smt_optim.core import Problem, ObjectiveConfig

from smt_optim.acquisition_functions.mpi import init_mpi


class MockSurrogate:
    def __init__(self, mu=0.0, s2=0.0, **kwargs):
        self.mu = mu
        self.s2 = s2

    def predict_values(self, x):
        # Return mocked mean
        return np.array([[self.mu]])

    def predict_variances(self, x):
        # Return mocked variance
        return np.array([[self.s2]])


class MockDataset:
    def export_as_dict(self):
        return {
            "x": np.array([[0.5]]),
            "obj": np.array([[1.0, 1.0]]),
            "rscv": np.array([0.0]),
            "fidelity": np.array([1]),
        }


class TestMPI(unittest.TestCase):
    def test_mpi_zero_variance(self):
        """
        Test that MPI properly handles zero variance by using the 1e-16 clipping.
        We mock the state and surrogate models to return specific predictions.
        """
        prob = Problem(
            [
                ObjectiveConfig([lambda x: x], MockSurrogate),
                ObjectiveConfig([lambda x: x], MockSurrogate),
            ],
            np.array([[0, 1]]),
        )
        prob.num_obj = 2

        # Mock dataset with one Pareto point at (1.0, 1.0)
        ds = MockDataset()

        state = State(prob)
        state.scaled_dataset = ds

        # We test a candidate point that is STRICTLY DOMINATED: mu = (2.0, 2.0)
        # Variance is 0.0, so the probability of improvement should be 0.0
        state.obj_models = [MockSurrogate(2.0, 0.0), MockSurrogate(2.0, 0.0)]

        acq_func = init_mpi(state)
        mpi_val = acq_func(np.array([[0.5]]))

        self.assertAlmostEqual(mpi_val.item(), 0.0, places=10)

        # We test a candidate point that STRICTLY DOMINATES the front: mu = (0.0, 0.0)
        # Variance is 0.0, so the probability of improvement should be 1.0
        state.obj_models = [MockSurrogate(0.0, 0.0), MockSurrogate(0.0, 0.0)]

        acq_func = init_mpi(state)
        mpi_val = acq_func(np.array([[0.5]]))

        self.assertAlmostEqual(mpi_val.item(), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
