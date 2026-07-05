import unittest
import numpy as np

from smt_optim.benchmarks.registry import get_problem
from smt_optim.benchmarks.base import PymooWrapper
from smt_optim.core import Driver, ObjectiveConfig, DriverConfig, Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies.mosego import MOSEGO
from smt_optim.acquisition_functions.multi_obj import init_bi_obj_ei

class TestMultiObjIntegration(unittest.TestCase):

    def test_mosego_zdt1(self):
        problem = get_problem('ZDT1')
        problem.set_dim(2)
        
        obj1_config = ObjectiveConfig(
            [problem.f1],
            type="minimize",
            surrogate=SmtAutoModel,
        )
        obj2_config = ObjectiveConfig(
            [problem.f2],
            type="minimize",
            surrogate=SmtAutoModel,
        )
        prob_definition = Problem(
            obj_configs=[obj1_config, obj2_config],
            design_space=problem.bounds,
        )
        opt_config = DriverConfig(
            max_iter=1,
            nt_init=5,
            verbose=False,
            scaling=True,
            seed=0,
        )
        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_func": init_bi_obj_ei, "n_start": 5, "sp_method": "SLSQP"},
        )
        # Should run without issues (logger crash etc.)
        state = driver.optimize()
        self.assertGreater(len(state.dataset.samples), 5)

    def test_pymoo_wrapper_multi_fidelity(self):
        # DTLZ5 has multi-fidelity objectives
        problem = get_problem('DTLZ5')
        pymoo_prob = PymooWrapper(problem)
        
        # Test evaluating points
        x = np.random.rand(5, problem.num_dim)
        out = {}
        # Should not raise TypeError: 'list' object is not callable
        pymoo_prob._evaluate(x, out)
        
        self.assertIn("F", out)
        self.assertEqual(out["F"].shape, (5, 2))

if __name__ == '__main__':
    unittest.main()
