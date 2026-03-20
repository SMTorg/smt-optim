import unittest
import numpy as np

from smt_optim.core import Sample, OptimizationDataset

from smt_optim.core.sample import sample_func

class TestSample(unittest.TestCase):

    def test_sample_initialization(self):
        x = np.array([1.0, 2.0])
        obj = np.array([0.5, 1.5])
        cstr = np.array([0.0])
        eval_time = np.array([0.01, 0.02, 0.03])

        s = Sample(
            x=x,
            fidelity=0,
            obj=obj,
            cstr=cstr,
            eval_time=eval_time,
        )

        self.assertTrue(np.array_equal(s.x, x))
        self.assertEqual(s.fidelity, 0)
        self.assertTrue(np.array_equal(s.obj, obj))
        self.assertTrue(np.array_equal(s.cstr, cstr))
        self.assertTrue(np.array_equal(s.eval_time, eval_time))
        self.assertIsInstance(s.metadata, dict)


class TestOptimizationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = OptimizationDataset()

        # Two objectives, one constraint
        self.s1 = Sample(
            x=np.array([0.0, 1.0]),
            fidelity=0,
            obj=np.array([1.0, 2.0]),
            cstr=np.array([10.0]),
            eval_time=None,
        )

        self.s2 = Sample(
            x=np.array([2.0, 3.0]),
            fidelity=0,
            obj=np.array([3.0, 4.0]),
            cstr=np.array([20.0]),
            eval_time=None,
        )

        self.s3 = Sample(
            x=np.array([4.0, 5.0]),
            fidelity=1,
            obj=np.array([5.0, 6.0]),
            cstr=np.array([30.0]),
            eval_time=None,
        )


    def test_add_initializes_dimensions(self):
        self.dataset.add(self.s1)

        self.assertEqual(self.dataset.num_obj, 2)
        self.assertEqual(self.dataset.num_cstr, 1)
        self.assertEqual(self.dataset.num_fidelity, 1)
        self.assertIn(0, self.dataset.fidelities)

    def test_add_dimension_mismatch_raises(self):
        self.dataset.add(self.s1)

        bad_sample = Sample(
            x=np.array([1.0, 2.0]),
            fidelity=0,
            obj=np.array([1.0]),   # wrong size
            cstr=np.array([10.0]),
            eval_time=None,
        )

        with self.assertRaises(Exception):
            self.dataset.add(bad_sample)

    def test_add_multiple_fidelities(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s3)

        self.assertEqual(self.dataset.num_fidelity, 2)
        self.assertCountEqual(self.dataset.fidelities, [0, 1])


    def test_get_by_fidelity(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s2)
        self.dataset.add(self.s3)

        lvl0 = self.dataset.get_by_fidelity(0)
        lvl1 = self.dataset.get_by_fidelity(1)

        self.assertEqual(len(lvl0), 2)
        self.assertEqual(len(lvl1), 1)
        self.assertTrue(all(s.fidelity == 0 for s in lvl0))
        self.assertTrue(all(s.fidelity == 1 for s in lvl1))


    def test_export_single_objective(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s2)

        result = self.dataset.export_data(idx=0, lvl=0)

        expected = np.array([
            [1.0],
            [3.0],
        ])

        self.assertTrue(np.array_equal(result, expected))

    def test_export_multiple_objectives(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s2)

        result = self.dataset.export_data(idx=[0, 1], lvl=0)

        expected = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        self.assertTrue(np.array_equal(result, expected))

    def test_export_constraint(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s2)

        # constraint index is num_obj + 0 = 2
        result = self.dataset.export_data(idx=2, lvl=0)

        expected = np.array([
            [10.0],
            [20.0],
        ])

        self.assertTrue(np.array_equal(result, expected))

    def test_export_mixed_obj_and_constraint(self):
        self.dataset.add(self.s1)
        self.dataset.add(self.s2)

        # objective 1 and constraint 0
        result = self.dataset.export_data(idx=[1, 2], lvl=0)

        expected = np.array([
            [2.0, 10.0],
            [4.0, 20.0],
        ])

        self.assertTrue(np.array_equal(result, expected))

    def test_export_empty_fidelity(self):
        self.dataset.add(self.s1)

        result = self.dataset.export_data(idx=0, lvl=999)

        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.size, 0)


class TestEvaluator(unittest.TestCase):


    def test_float_output(self):

        def float_output(x):
            return 0.0

        value, time = sample_func(np.array([0.0]), float_output)

        self.assertEqual(value, 0.0)


    def test_array_output(self):

        def array_1d_output(x):
            return np.array([0.0])

        value, time = sample_func(np.array([0.0]), array_1d_output)
        self.assertEqual(value, 0.0)

        def array_2d_output(x):
            return np.array([[0.0]])

        value, time = sample_func(np.array([0.0]), array_2d_output)
        self.assertEqual(value, 0.0)



if __name__ == "__main__":
    unittest.main()