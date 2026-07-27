import unittest
import numpy as np

from smt_optim.acquisition_functions.composite_expected_improvement import init_bi_obj_composite_ei


class DummyModel:
    def __init__(self, value, variance):
        self.val = value
        self.var = variance

    def predict_values(self, x_pred):
        return np.array([self.val])

    def predict_variances(self, x_pred):
        return np.array([self.var])


class DummyDataset:
    def __init__(self, data):
        self.data = np.array(data)

    def export_data(self, indices, column):
        return self.data


class DummyState:
    def __init__(self, model_0, model_1, dataset):
        self.obj_models = [model_0, model_1]
        self.scaled_dataset = dataset


class TestCompositeExpectedImprovementNoMocks(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset with two training points
        # If phi(y) = y0 + y1:
        # Point 1: [1.0, 1.0] -> phi = 2.0
        # Point 2: [2.0, 2.0] -> phi = 4.0
        # f_min will be min(2.0, 4.0) = 2.0
        self.dataset = DummyDataset([[1.0, 1.0], [2.0, 2.0]])
        self.phi = lambda y: np.sum(y)

    def test_composite_ei_zero_variance_no_improvement(self):
        """Test that if variance is 0 and mean does not improve, EI is 0."""
        # Predictions: mean = [1.5, 1.5] -> phi = 3.0 (worse than f_min of 2.0)
        # Variance: 0.0
        model_0 = DummyModel(value=1.5, variance=0.0)
        model_1 = DummyModel(value=1.5, variance=0.0)
        state = DummyState(model_0, model_1, self.dataset)

        kwargs = {"phi": self.phi, "n_accuracy": 100}
        composite_ei = init_bi_obj_composite_ei(state, kwargs)

        # Act & Assert
        ei_val = composite_ei(np.array([0.0]))
        self.assertEqual(ei_val, 0.0)

    def test_composite_ei_zero_variance_with_improvement(self):
        """Test deterministic improvement when variance is 0."""
        # Predictions: mean = [0.5, 0.5] -> phi = 1.0 (better than f_min of 2.0)
        # Variance: 0.0
        # Expected Improvement: f_min - phi = 2.0 - 1.0 = 1.0
        model_0 = DummyModel(value=0.5, variance=0.0)
        model_1 = DummyModel(value=0.5, variance=0.0)
        state = DummyState(model_0, model_1, self.dataset)

        kwargs = {"phi": self.phi, "n_accuracy": 10}
        composite_ei = init_bi_obj_composite_ei(state, kwargs)

        # Act & Assert
        ei_val = composite_ei(np.array([0.0]))
        self.assertAlmostEqual(ei_val, 1.0, places=5)

    def test_composite_ei_with_variance(self):
        """Test Monte Carlo integration with variance (using a random seed)."""
        model_0 = DummyModel(value=1.0, variance=1.0)
        model_1 = DummyModel(value=1.0, variance=1.0)
        state = DummyState(model_0, model_1, self.dataset)

        kwargs = {"phi": self.phi, "n_accuracy": 1000}
        composite_ei = init_bi_obj_composite_ei(state, kwargs)

        # Set seed for reproducibility
        np.random.seed(42)
        ei_val_1 = composite_ei(np.array([0.0]))

        # Re-seed to verify determinism
        np.random.seed(42)
        ei_val_2 = composite_ei(np.array([0.0]))

        self.assertEqual(ei_val_1, ei_val_2)
        self.assertGreater(ei_val_1, 0.0)


if __name__ == "__main__":
    unittest.main()