import unittest
import numpy as np
from scipy import stats

from smt_optim.core import ObjectiveConfig, Problem
from smt_optim.core.state import State
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_functions import log_ei, q_log_ei
from smt_optim.acquisition_strategies import qLogEI


def normal_samples(batch_size, n_samples=2**14, seed=0):
    sampler = stats.qmc.LatinHypercube(d=batch_size, seed=seed)
    return stats.norm.ppf(sampler.random(n_samples))


class TestQLogExpectedImprovement(unittest.TestCase):
    def test_q_log_ei(self):

        mu = np.array([[1.0], [1.0]])
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        f_min = 0.5
        z = normal_samples(2)

        value = q_log_ei(mu, cov, f_min, z)

        # test output type
        self.assertIsInstance(value, float)

        # test that the same samples give the same value
        self.assertEqual(value, q_log_ei(mu, cov, f_min, z))

    def test_q_log_ei_corresponds_to_log_ei(self):

        mu = np.array([[1.0]])
        cov = np.array([[1.0]])
        f_min = 0.5
        single = q_log_ei(mu, cov, f_min, normal_samples(1))

        # with one point, q_log_ei corresponds to log_ei
        self.assertAlmostEqual(single, log_ei(1.0, 1.0, f_min), delta=1e-3)

        # two identical points are worth one point
        mu = np.array([[1.0], [1.0]])
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(
            q_log_ei(mu, cov, f_min, normal_samples(2)), single, delta=0.01
        )

    def test_q_log_ei_no_uncertainty(self):

        mu = np.array([[-1.5], [-0.5]])
        cov = np.zeros((2, 2))
        f_min = 0.5

        # the best point improves on f_min by 2
        value = q_log_ei(mu, cov, f_min, normal_samples(2))
        self.assertAlmostEqual(value, np.log(2.0), delta=1e-3)

    def test_q_log_ei_correlation(self):

        mu = np.array([[1.0], [1.0]])
        f_min = 0.5
        z = normal_samples(2)

        correlated = np.array([[1.0, 0.9], [0.9, 1.0]])
        uncorrelated = np.array([[1.0, 0.0], [0.0, 1.0]])

        # less correlated points have a higher value
        self.assertGreater(
            q_log_ei(mu, uncorrelated, f_min, z), q_log_ei(mu, correlated, f_min, z)
        )

    def test_q_log_ei_no_improvement(self):

        cov = np.zeros((2, 2))
        f_min = 0.5
        z = normal_samples(2)

        far = q_log_ei(np.array([[100.0], [100.0]]), cov, f_min, z)
        farther = q_log_ei(np.array([[101.0], [101.0]]), cov, f_min, z)

        # the value stays finite and can still rank the batches
        self.assertTrue(np.isfinite(far))
        self.assertGreater(far, farther)


class DummyModel:
    def predict_values(self, x):
        pass

    def predict_covariances(self, x):
        pass


class DummyModelNoCovariances:
    def predict_values(self, x):
        pass


def sphere(x):
    return float(np.sum(x**2))


def build_state(num_obj=1, num_fidelity=1, model=None):
    obj_configs = [
        ObjectiveConfig(objective=[sphere] * num_fidelity, surrogate=SmtAutoModel)
        for _ in range(num_obj)
    ]
    problem = Problem(
        obj_configs=obj_configs,
        design_space=np.array([[0.0, 1.0], [0.0, 1.0]]),
        costs=[1.0] * num_fidelity,
    )
    state = State(problem)
    state.obj_models = [model]
    return state


class TestQLogEIValidateConfig(unittest.TestCase):
    def test_valid_config(self):
        state = build_state(model=DummyModel())
        qLogEI(state).validate_config(state)

    def test_multi_objective(self):
        state = build_state(num_obj=2, model=DummyModel())
        with self.assertRaisesRegex(Exception, "Multi-objective"):
            qLogEI(state).validate_config(state)

    def test_multi_fidelity(self):
        state = build_state(num_fidelity=2, model=DummyModel())
        with self.assertRaisesRegex(Exception, "Multi-fidelity"):
            qLogEI(state).validate_config(state)

    def test_missing_covariances(self):
        state = build_state(model=DummyModelNoCovariances())
        with self.assertRaises(TypeError):
            qLogEI(state).validate_config(state)


if __name__ == "__main__":
    unittest.main()
