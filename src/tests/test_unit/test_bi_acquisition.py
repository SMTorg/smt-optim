import unittest

import numpy as np

from smt_optim.acquisition_functions import init_bi_acquisition, negate


class TestBiAcquisition(unittest.TestCase):
    def test_init_bi_acquisition(self):

        x = np.array([[1.0], [2.0]])

        def acq_init_1(state):
            return lambda x: x

        def acq_init_2(state):
            return lambda x: -x[:, 0] * state

        bi_acquisition = init_bi_acquisition(10.0, acq_init_1, acq_init_2)

        # columns follow the order of the acquisition functions, with the values unchanged
        np.testing.assert_array_equal(
            bi_acquisition(x), np.array([[1.0, -10.0], [2.0, -20.0]])
        )

    def test_negate(self):

        x = np.array([[1.0], [2.0]])
        calls = []

        def acq_init(state):
            calls.append(state)
            return lambda x: state * x

        neg_acq = negate(acq_init)(3.0)

        np.testing.assert_array_equal(neg_acq(x), -3.0 * x)

        # the acquisition function is only initialized once
        neg_acq(x)
        self.assertEqual(calls, [3.0])

    def test_init_bi_acquisition_invalid_shape(self):

        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        invalid_acq_funcs = [
            lambda x: 1.0,  # not vectorized
            lambda x: np.array([[1.0]]),  # not vectorized
            lambda x: x,  # more than one value per point
        ]

        for acq in invalid_acq_funcs:
            bi_acquisition = init_bi_acquisition(None, lambda s: acq, lambda s: acq)

            with self.assertRaises(ValueError):
                bi_acquisition(x)


if __name__ == "__main__":
    unittest.main()
