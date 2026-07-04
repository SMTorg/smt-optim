import unittest
import numpy as np
from smt_optim.acquisition_functions.multi_obj import init_mpi, init_bi_obj_pi, init_bi_obj_ei_cf, init_bi_obj_ei
from smt_optim.utils.multi_obj import get_pf_from_dataset

class MockModel:
    def __init__(self, val, var):
        self.val = val
        self.var = var
    def predict_values(self, x):
        return np.ones((len(x), 1)) * self.val
    def predict_variances(self, x):
        return np.ones((len(x), 1)) * self.var

class MockProblem:
    def __init__(self):
        self.num_obj = 2

class MockDataset:
    def __init__(self):
        pass
    def export_as_dict(self):
        return {
            "obj": np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]]),
            "rscv": np.array([0.0, 0.0, 0.0]),
            "fidelity": np.array([0, 0, 0]),
            "x": np.array([[0.1], [0.9], [0.5]]),
        }

class MockState:
    def __init__(self):
        self.obj_models = [MockModel(0.5, 0.1), MockModel(0.5, 0.1)]
        self.problem = MockProblem()
        self.scaled_dataset = MockDataset()

class TestMOAcqFunctions(unittest.TestCase):
    def test_all_acq(self):
        state = MockState()
        x_test = np.array([[0.5]])
        
        mpi = init_mpi(state)
        val_mpi = mpi(x_test)
        self.assertTrue(isinstance(val_mpi, (float, np.ndarray)))
        
        pi = init_bi_obj_pi(state)
        val_pi = pi(x_test)
        self.assertTrue(isinstance(val_pi, (float, np.ndarray)))
        
        ei_ana = init_bi_obj_ei(state)
        val_ei_ana = ei_ana(x_test)
        self.assertTrue(isinstance(val_ei_ana, (float, np.ndarray)))
        
        def mock_phi(y):
            return np.sum(y, axis=-1)
        ei_cf = init_bi_obj_ei_cf(state, {"phi": mock_phi})
        val_ei_cf = ei_cf(x_test)
        self.assertTrue(isinstance(val_ei_cf, (float, np.ndarray)))
        
        pf = get_pf_from_dataset(state.scaled_dataset)
        self.assertTrue(len(pf) > 0)

if __name__ == "__main__":
    unittest.main()
