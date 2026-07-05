import unittest
import numpy as np

from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.benchmarks.multi_obj.zdt import ZDT1
from smt_optim.utils.multi_obj import get_pf_from_dataset, hypervolume_2d
from smt_optim.acquisition_strategies.biego import BiEGO
from smt_optim.acquisition_strategies.mosego import MOSEGO

class TestMultiObjectiveConvergence(unittest.TestCase):
    def test_zdt1_mosego_hypervolume_growth(self):
        """Test that MOSEGO improves the hypervolume of the Pareto front on ZDT1."""
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj1 = ObjectiveConfig(objective=[zdt1.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[zdt1.f2], surrogate=SmtAutoModel)

        problem = Problem(obj_configs=[obj1, obj2], design_space=zdt1.bounds)

        opt_config = DriverConfig(max_iter=5, nt_init=5, seed=42)

        driver = Driver(problem=problem, config=opt_config, strategy=MOSEGO)
        
        # Initial DoE hypervolume
        driver.setup()
        initial_dataset = driver.state.scaled_dataset
        initial_pf = get_pf_from_dataset(initial_dataset)
        ref_point = np.array([2.0, 2.0])
        initial_hv = hypervolume_2d(initial_pf, ref_point)

        # Optimize for a few iterations
        state = driver.optimize()
        final_pf = get_pf_from_dataset(state.scaled_dataset)
        final_hv = hypervolume_2d(final_pf, ref_point)

        from pymoo.indicators.igd_plus import IGDPlus
        x1 = np.linspace(0, 1, 100)
        true_pf = np.array([x1, 1 - np.sqrt(x1)]).T
        igd = IGDPlus(true_pf)
        
        initial_igd = igd.do(initial_pf)
        final_igd = igd.do(final_pf)

        # Hypervolume should increase or remain the same
        self.assertGreaterEqual(final_hv, initial_hv)
        # IGD+ should decrease or remain the same
        self.assertLessEqual(final_igd, initial_igd)

    def test_zdt1_biego_hypervolume_growth(self):
        """Test that BiEGO improves the hypervolume of the Pareto front on ZDT1."""
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj1 = ObjectiveConfig(objective=[zdt1.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[zdt1.f2], surrogate=SmtAutoModel)

        problem = Problem(obj_configs=[obj1, obj2], design_space=zdt1.bounds)

        # Provide enough max_iter to cover Min(f1), Min(f2), and Bi-objective phases
        # min_max_calls is nt_init by default (5), so BiEGO needs 10 iters to reach bi-objective phase.
        opt_config = DriverConfig(max_iter=15, nt_init=5, seed=42)

        driver = Driver(problem=problem, config=opt_config, strategy=BiEGO)
        
        # Initial DoE hypervolume
        driver.setup()
        initial_dataset = driver.state.scaled_dataset
        initial_pf = get_pf_from_dataset(initial_dataset)
        ref_point = np.array([2.0, 2.0])
        initial_hv = hypervolume_2d(initial_pf, ref_point)

        # Optimize
        state = driver.optimize()
        final_pf = get_pf_from_dataset(state.scaled_dataset)
        final_hv = hypervolume_2d(final_pf, ref_point)

        from pymoo.indicators.igd_plus import IGDPlus
        from smt_optim.utils.multi_obj import get_pareto_front
        # We need a true pareto front or a reference front for IGD+.
        # We'll just generate the true ZDT1 front analytically.
        x1 = np.linspace(0, 1, 100)
        true_pf = np.array([x1, 1 - np.sqrt(x1)]).T
        igd = IGDPlus(true_pf)
        
        initial_igd = igd.do(initial_pf)
        final_igd = igd.do(final_pf)

        # Hypervolume should increase or remain the same
        self.assertGreaterEqual(final_hv, initial_hv)
        # IGD+ should decrease or remain the same
        self.assertLessEqual(final_igd, initial_igd)

if __name__ == '__main__':
    unittest.main()
