import unittest

import warnings
import numpy as np

import smt.design_space as ds

from smt_optim.core.driver import safe_descale, infill_not_in_xt



class TestSafeDescale(unittest.TestCase):

    def test_cont_design_space(self):
        # ------- single continuous variable -------
        class DummyProblem:
            def __init__(self):
                self.design_space = ds.DesignSpace([
                    ds.FloatVariable(0, 1)
                ])

        class DummyState:
            def __init__(self):
                self.x_factor = np.array([1]),
                self.x_step = np.array([0]),
                self.problem = DummyProblem()

        state = DummyState()

        x_raw = np.array([[0.5]])

        x_new = safe_descale(x_raw, state)

        # output shape matches input shape
        self.assertEqual(x_raw.shape[0], x_new.shape[0])
        self.assertEqual(x_raw.shape[1], x_new.shape[1])

        self.assertEqual(x_raw.item(), x_new.item())

        # ------- multiple continuous variables -------
        class DummyProblem:
            def __init__(self):
                self.design_space = ds.DesignSpace([
                    ds.FloatVariable(-5, 5),
                    ds.FloatVariable(-5, 5)
                ])

        class DummyState:
            def __init__(self):
                self.x_factor = np.array([10]),
                self.x_step = np.array([-5]),
                self.problem = DummyProblem()

        state = DummyState()

        x_raw = np.array([
            [0.1, 1.0],
            [0.5, 1.1],
        ])

        with warnings.catch_warnings(record=True) as w:
            x_new = safe_descale(x_raw, state)

            # verify a warning happened
            self.assertEqual(len(w), 1)

            # output shape matches input shape
            self.assertEqual(x_raw.shape[0], x_new.shape[0])
            self.assertEqual(x_raw.shape[1], x_new.shape[1])

            self.assertEqual(x_new[0, 1], 5)
            self.assertEqual(x_new[1, 1], 5)

    def test_mix_design_space(self):

        # ------- single mixed variable -------
        class DummyProblem:
            def __init__(self):
                self.design_space = ds.DesignSpace([
                    ds.CategoricalVariable([0, 1, 2])
                ])

        class DummyState:
            def __init__(self):
                self.x_factor = np.array([1]),
                self.x_step = np.array([0]),
                self.problem = DummyProblem()

        state = DummyState()

        x_raw = np.array([[1]])

        x_new = safe_descale(x_raw, state)

        # output shape matches input shape
        self.assertEqual(x_raw.shape[0], x_new.shape[0])
        self.assertEqual(x_raw.shape[1], x_new.shape[1])

        self.assertEqual(x_raw.item(), x_new.item())

        # ------- multiple mixed variables -------
        class DummyProblem:
            def __init__(self):
                self.design_space = ds.DesignSpace([
                    ds.CategoricalVariable([0, 1]),
                    ds.IntegerVariable(-5, 5),
                    ds.FloatVariable(-5, 5)
                ])

        class DummyState:
            def __init__(self):
                self.x_factor = np.array([1, 1, 10]),
                self.x_step = np.array([0, 0, -5]),
                self.problem = DummyProblem()

        state = DummyState()

        x_raw = np.array([
            [0.1, 0.6, 0.5],    # cat. and int. variables are invalid
            [2, -6, 1.1],       # all variables are out of bounds
            [0, 1, 0.8]         # all variables are valid
        ])

        # safe descaled solution:
        # [ 0.  1.  0.]
        # [ 1. -5.  5.]
        # [ 0.  1.  3.]

        with warnings.catch_warnings(record=True) as w:
            x_new = safe_descale(x_raw, state)

            # verify a single warning happened
            self.assertEqual(len(w), 1)

            # output shape matches input shape
            self.assertEqual(x_raw.shape[0], x_new.shape[0])
            self.assertEqual(x_raw.shape[1], x_new.shape[1])

            # verify first row
            self.assertEqual(x_new[0, 0], 0)
            self.assertEqual(x_new[1, 0], 1)

            # verify second row
            self.assertEqual(x_new[0, 1], 1)
            self.assertEqual(x_new[1, 1], -5)
            self.assertEqual(x_new[1, 2], 5)



class DummyDataset:
    def __init__(self, x, fidelity):
        self._x = x
        self._fidelity = fidelity

    def export_as_dict(self):
        return {
            "x": self._x,
            "fidelity": self._fidelity
        }

class DummyState:
    def __init__(self, x, fidelity):
        self.dataset = DummyDataset(x, fidelity)

class TestInfillNotInXt(unittest.TestCase):


    def test_infill_not_in_xt(self):
        # Training data
        xt = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0]
        ])

        fidelity = np.array([
            0,
            0,
            1,
            1
        ])

        state = DummyState(xt, fidelity)

        # ------- Case 1: valid infill (no overlap) -------
        infills = [
            np.array([[0.5, 0.5]]),  # lvl 0
            np.array([[3.0, 3.0]])  # lvl 1
        ]

        # Should not raise
        infill_not_in_xt(infills, state)

        # ------- Case 2: duplicate at same fidelity -------
        infills = [
            np.array([[1.0, 1.0]]),  # duplicate at lvl 0
            None
        ]

        with self.assertRaises(Exception) as cm:
            infill_not_in_xt(infills, state)

        self.assertIn("Infill point already in training data", str(cm.exception))

        # ------- Case 3: duplicate but different fidelity -> allowed -------
        infills = [
            None,
            np.array([[1.0, 1.0]])  # exists in xt but at lvl 0, not lvl 1
        ]

        # Should NOT raise
        infill_not_in_xt(infills, state)

        # ------- Case 4: multiple infill points, one invalid -------
        infills = [
            np.array([
                [0.2, 0.2],
                [0.0, 0.0]  # duplicate
            ]),
            None
        ]

        with self.assertRaises(Exception):
            infill_not_in_xt(infills, state)

        # ------- Case 5: near-duplicate (tolerance check) -------
        infills = [
            np.array([[1.0 + 1e-10, 1.0]]),  # within the 1e-8 tolerance
            None
        ]

        with self.assertRaises(Exception):
            infill_not_in_xt(infills, state)