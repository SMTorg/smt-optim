import unittest
import numpy as np

from smt_optim.benchmarks.multiobj.zdt_mf import DTLZ5, ZDT1, ZDT2, ZDT3
from smt_optim.benchmarks.base import PymooWrapper


class TestDTLZ5(unittest.TestCase):
    def test_defaults_before_set_dim(self):
        prob = DTLZ5()

        self.assertEqual(prob.name, "DTLZ5")
        self.assertEqual(prob.num_dim, "variable")
        self.assertEqual(prob.num_obj, 2)
        self.assertEqual(prob.num_cstr, 1)
        self.assertEqual(prob.num_fidelity, 2)
        self.assertIn("n_variable", prob.tags)
        np.testing.assert_allclose(prob.bounds, np.array([[0, 1]]))

    def test_set_dim_updates_bounds(self):
        prob = DTLZ5()
        prob.set_dim(4)

        self.assertEqual(prob.num_dim, 4)
        np.testing.assert_allclose(prob.bounds, np.tile([0, 1], (4, 1)))

    def test_u_function(self):
        prob = DTLZ5()
        np.testing.assert_allclose(prob.u(np.array([0.5, 0.5])), 0.0)
        np.testing.assert_allclose(prob.u(np.array([0.0, 0.0])), 0.5)
        np.testing.assert_allclose(prob.u(np.array([])), 0.0)

    def test_high_fidelity_functions_at_origin(self):
        prob = DTLZ5()
        prob.set_dim(5)
        x = np.zeros(5)

        np.testing.assert_allclose(prob.f1(x), 1.5)
        np.testing.assert_allclose(prob.f2(x), 0.0, atol=1e-12)
        np.testing.assert_allclose(prob.g(x), 1.0)

    def test_low_fidelity_functions_at_origin(self):
        prob = DTLZ5()
        prob.set_dim(5)
        x = np.zeros(5)

        np.testing.assert_allclose(prob.f1_lf(x), 1.4)
        np.testing.assert_allclose(prob.f2_lf(x), 0.0, atol=1e-12)
        # note: g_lf does not subtract the 0.5 offset that g does
        np.testing.assert_allclose(prob.g_lf(x), 1.4)

    def test_functions_generic_point(self):
        prob = DTLZ5()
        prob.set_dim(4)
        x = np.array([0.3, 0.1, 0.2, 0.4])
        u = np.sum((x[3:] - 0.5) ** 2)

        expected_f1 = (1 + u) * np.cos(np.pi / 2 * x[0])
        expected_f2 = (1 + u) * np.sin(np.pi / 2 * x[0])
        expected_f1_lf = (1 + 0.8 * u) * np.cos(np.pi / 2 * x[0])
        expected_f2_lf = (1 + 1.1 * u) * np.sin(np.pi / 2 * x[0])

        np.testing.assert_allclose(prob.f1(x), expected_f1)
        np.testing.assert_allclose(prob.f2(x), expected_f2)
        np.testing.assert_allclose(prob.g(x), expected_f1 - 0.5)

        np.testing.assert_allclose(prob.f1_lf(x), expected_f1_lf)
        np.testing.assert_allclose(prob.f2_lf(x), expected_f2_lf)
        np.testing.assert_allclose(prob.g_lf(x), expected_f1_lf)

    def test_objective_and_constraint_lists(self):
        prob = DTLZ5()
        prob.set_dim(5)
        x = np.zeros(5)

        self.assertEqual(len(prob.objective), prob.num_obj)
        self.assertEqual(len(prob.constraints), prob.num_cstr)

        for i, funcs in enumerate([[prob.f1_lf, prob.f1], [prob.f2_lf, prob.f2]]):
            self.assertEqual(len(prob.objective[i]), prob.num_fidelity)
            np.testing.assert_allclose(prob.objective[i][0](x), funcs[0](x))
            np.testing.assert_allclose(prob.objective[i][-1](x), funcs[1](x))

        np.testing.assert_allclose(prob.constraints[0][0](x), prob.g_lf(x))
        np.testing.assert_allclose(prob.constraints[0][-1](x), prob.g(x))

    def test_pymoo_wrapper_uses_high_fidelity(self):
        prob = DTLZ5()
        prob.set_dim(5)
        wrapper = PymooWrapper(prob)

        self.assertEqual(wrapper.n_var, 5)
        self.assertEqual(wrapper.n_obj, 2)
        self.assertEqual(wrapper.n_ieq_constr, 1)

        X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.5, 0.5]])
        out = {}
        wrapper._evaluate(X, out)

        expected_F = np.array([[prob.f1(x), prob.f2(x)] for x in X])
        expected_G = np.array([[prob.g(x)] for x in X])

        np.testing.assert_allclose(out["F"], expected_F, atol=1e-12)
        np.testing.assert_allclose(out["G"], expected_G, atol=1e-12)


class _ZDTMultiFidelityMixin:
    """Shared behavior/formula checks for the ZDT1/ZDT2/ZDT3 multi-fidelity classes."""

    cls = None  # set by subclass
    dim = 3

    def make(self, dim=None):
        prob = self.cls()
        prob.set_dim(dim or self.dim)
        return prob

    def test_metadata(self):
        prob = self.cls()

        self.assertEqual(prob.name, self.cls.__name__)
        self.assertEqual(prob.num_obj, 2)
        self.assertEqual(prob.num_cstr, 0)
        self.assertEqual(prob.num_fidelity, 2)
        self.assertIsNone(prob.constraints)
        self.assertIn("n_variable", prob.tags)

    def test_default_dim_is_30_but_bounds_not_yet_expanded(self):
        # NOTE: unlike DTLZ5 (num_dim starts as "variable"), these classes are
        # preset to num_dim=30 without calling set_dim(), while bounds is
        # still the un-expanded single row. set_dim() must be called before
        # the bounds array actually matches num_dim.
        prob = self.cls()
        self.assertEqual(prob.num_dim, 30)
        self.assertEqual(prob.bounds.shape, (1, 2))

    def test_set_dim_expands_bounds(self):
        prob = self.make(dim=5)
        self.assertEqual(prob.num_dim, 5)
        np.testing.assert_allclose(prob.bounds, np.tile([0, 1], (5, 1)))

    def test_g_and_g_lf_are_identical(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])
        np.testing.assert_allclose(prob.g(x), prob.g_lf(x))

        expected_g = 1 + 9 * np.sum(x[1:]) / (prob.num_dim - 1)
        np.testing.assert_allclose(prob.g(x), expected_g)

    def test_f1_and_f2_at_origin(self):
        prob = self.make()
        x = np.zeros(3)

        np.testing.assert_allclose(prob.f1(x), 0.0)
        np.testing.assert_allclose(prob.g(x), 1.0)
        np.testing.assert_allclose(prob.f2(x), prob.g(x) * prob.h(prob.f1(x), prob.g(x)))

    def test_f2_matches_formula_generic_point(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])

        g_val = prob.g(x)
        f1_val = prob.f1(x)
        expected_f2 = g_val * prob.h(f1_val, g_val)
        np.testing.assert_allclose(prob.f2(x), expected_f2)

        g_lf_val = prob.g_lf(x)
        f1_lf_val = prob.f1_lf(x)
        h_lf_val = prob.h(f1_lf_val, g_lf_val)
        np.testing.assert_allclose(prob.f2_lf(x), self.expected_f2_lf(g_lf_val, h_lf_val))

    def expected_f2_lf(self, g_val, h_val):
        raise NotImplementedError

    def test_objective_list_structure(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])

        self.assertEqual(len(prob.objective), prob.num_obj)
        for i, funcs in enumerate([[prob.f1_lf, prob.f1], [prob.f2_lf, prob.f2]]):
            self.assertEqual(len(prob.objective[i]), prob.num_fidelity)
            np.testing.assert_allclose(prob.objective[i][0](x), funcs[0](x))
            np.testing.assert_allclose(prob.objective[i][-1](x), funcs[1](x))

    def test_pymoo_wrapper_uses_high_fidelity(self):
        prob = self.make(dim=3)
        wrapper = PymooWrapper(prob)

        self.assertEqual(wrapper.n_var, 3)
        self.assertEqual(wrapper.n_obj, 2)
        self.assertEqual(wrapper.n_ieq_constr, 0)

        X = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.3]])
        out = {}
        wrapper._evaluate(X, out)

        expected_F = np.array([[prob.f1(x), prob.f2(x)] for x in X])
        np.testing.assert_allclose(out["F"], expected_F, atol=1e-12)
        self.assertNotIn("G", out)


class TestZDT1MultiFidelity(_ZDTMultiFidelityMixin, unittest.TestCase):
    cls = ZDT1

    def test_f1_lf_is_linear_transform_of_f1(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])
        np.testing.assert_allclose(prob.f1_lf(x), 0.9 * prob.f1(x) + 0.1)

    def expected_f2_lf(self, g_val, h_val):
        return (0.8 * g_val - 0.2) * (1.2 * h_val + 0.2)


class TestZDT2MultiFidelity(_ZDTMultiFidelityMixin, unittest.TestCase):
    cls = ZDT2

    def test_f1_lf_is_linear_transform_of_f1(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])
        np.testing.assert_allclose(prob.f1_lf(x), 0.8 * prob.f1(x) + 0.2)

    def expected_f2_lf(self, g_val, h_val):
        return (0.9 * g_val + 0.2) * (1.1 * h_val - 0.2)


class TestZDT3MultiFidelity(_ZDTMultiFidelityMixin, unittest.TestCase):
    cls = ZDT3

    def test_f1_lf_is_linear_transform_of_f1(self):
        prob = self.make()
        x = np.array([0.5, 0.2, 0.3])
        np.testing.assert_allclose(prob.f1_lf(x), 0.75 * prob.f1(x) + 0.25)

    def test_h_includes_discontinuous_sine_term(self):
        prob = self.make()
        f1, g = 0.4, 2.0
        expected_h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        np.testing.assert_allclose(prob.h(f1, g), expected_h)

    def expected_f2_lf(self, g_val, h_val):
        return g_val * (1.25 * h_val - 0.25)


if __name__ == "__main__":
    unittest.main()
