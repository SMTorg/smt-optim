import unittest
import numpy as np

from smt_optim.acquisition_strategies.biego import (
    SingleObjectiveProduct,
    SingleObjectiveNormalized,
)
from smt_optim.acquisition_functions.multi_obj import init_mpi


class MockModel:
    def __init__(self, val, var):
        self.val = val
        self.var = var

    def predict_values(self, x):
        return np.array([[self.val]])

    def predict_variances(self, x):
        return np.array([[self.var]])


class MockProblem:
    def __init__(self):
        self.num_obj = 2


class MockDataset:
    def __init__(self):
        self.points = np.array([[0.5, 0.5]])

    def export_as_dict(self):
        return {
            "obj": np.array([[0.2, 0.2], [0.8, 0.8]]),
            "rscv": np.array([0.0, 0.0]),
            "fidelity": np.array([0, 0]),
            "x": np.array([[0.1], [0.9]]),
        }


class MockState:
    def __init__(self):
        self.obj_models = [MockModel(0.1, 0.0), MockModel(0.1, 0.0)]
        self.problem = MockProblem()
        self.scaled_dataset = MockDataset()


class TestMultiObjectiveFixes(unittest.TestCase):
    def test_vectorization_single_objective_product(self):
        # Shape: (2,)
        x_1d = np.array([0.5, 0.3])
        r = np.array([1.0, 1.0])
        # r - x = [0.5, 0.7] => max(0)^2 = [0.25, 0.49] => prod = 0.1225 => return -0.1225
        val_1d = SingleObjectiveProduct(x_1d, r)
        self.assertAlmostEqual(val_1d, -0.1225)

        # Shape: (N, M, 2)
        x_3d = np.array([[[0.5, 0.3], [1.2, 0.9]], [[1.1, 1.1], [0.8, 0.2]]])
        val_3d = SingleObjectiveProduct(x_3d, r)
        expected = np.array(
            [
                [-0.1225, 0.0],
                [0.0, -0.0256],  # [0.8, 0.2] => [0.2, 0.8] => 0.04 * 0.64 = 0.0256
            ]
        )
        np.testing.assert_allclose(val_3d, expected)

    def test_vectorization_single_objective_normalized(self):
        # Shape: (2,)
        x_1d = np.array([0.5, 0.3])
        r = np.array([0.1, 0.1])
        w = np.array([2.0, 1.0])

        val_1d = SingleObjectiveNormalized(x_1d, r, w)
        # max((0.5-0.1)/2, (0.3-0.1)/1) = max(0.2, 0.2) = 0.2
        self.assertAlmostEqual(val_1d, 0.2)

        # Shape: (N, 2)
        x_2d = np.array([[0.5, 0.3], [0.2, 0.9]])
        val_2d = SingleObjectiveNormalized(x_2d, r, w)
        # row2: max((0.2-0.1)/2, (0.9-0.1)/1) = max(0.05, 0.8) = 0.8
        expected = np.array([0.2, 0.8])
        np.testing.assert_allclose(val_2d, expected)

    def test_mpi_zero_variance(self):
        state = MockState()
        # Initialize MPI
        mpi_func = init_mpi(state)
        # Point to predict
        x_test = np.array([[0.5]])
        # Since variances are 0.0, MPI should return 0.0 to prevent infinite loop
        val = mpi_func(x_test)
        self.assertEqual(val, 0.0)

    def test_ei_cf_constraint_filtering(self):
        from smt_optim.acquisition_functions.multi_obj import init_bi_obj_ei_cf

        state = MockState()
        # Create a mock dataset with one feasible and one infeasible point
        # [0.2, 0.2] has rscv = 1.0 (infeasible)
        # [0.8, 0.8] has rscv = 0.0 (feasible)
        state.scaled_dataset.export_as_dict = lambda: {
            "obj": np.array([[0.2, 0.2], [0.8, 0.8]]),
            "rscv": np.array([1.0, 0.0]),
            "fidelity": np.array([0, 0]),
            "x": np.array([[0.1], [0.9]]),
        }

        # phi is just sum of objectives
        def mock_phi(y):
            return np.sum(y, axis=-1)

        kwargs = {"phi": mock_phi, "n_accuracy": 10}

        # init_bi_obj_ei_cf should find f_min = mock_phi([0.8, 0.8]) = 1.6
        # rather than mock_phi([0.2, 0.2]) = 0.4

        # Since I can't easily extract f_min from the closure without calling it,
        # I'll call it with a point. If f_min = 1.6, and predictions are 0.1, 0.1
        # Then phi(y) = 0.2. max(f_min - phi(y), 0) = max(1.6 - 0.2, 0) = 1.4
        # If f_min was 0.4, max(0.4 - 0.2, 0) = 0.2
        ei_cf_func = init_bi_obj_ei_cf(state, kwargs)

        val = ei_cf_func(np.array([[0.5]]))
        self.assertAlmostEqual(val[0][0], 1.4)


if __name__ == "__main__":
    unittest.main()
