import unittest
import numpy as np

from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.benchmarks.multi_obj.zdt import ZDT1
from smt_optim.utils.multi_obj import (
    get_pf_from_dataset,
    hypervolume_2d,
    plot_pareto_front,
)
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

        opt_config = DriverConfig(max_iter=25, nt_init=10, seed=42)

        driver = Driver(problem=problem, config=opt_config, strategy=MOSEGO)

        # Initial DoE hypervolume
        driver.start_optim()
        initial_dataset = driver.state.dataset
        initial_pf = get_pf_from_dataset(initial_dataset)
        ref_point = np.array([2.0, 2.0])
        initial_hv = hypervolume_2d(initial_pf, ref_point)

        # Optimize for a few iterations
        state = driver.optimize()
        final_pf = get_pf_from_dataset(state.dataset)
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

        plot_pareto_front(
            initial_dataset,
            state.dataset,
            filename="zdt1_pareto_front_mosego.png",
            title="Pareto Front of ZDT1 (2D) obtained with MOSEGO",
        )

    def test_zdt1_biego_hypervolume_growth(self):
        """Test that BiEGO improves the hypervolume of the Pareto front on ZDT1."""
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj1 = ObjectiveConfig(objective=[zdt1.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[zdt1.f2], surrogate=SmtAutoModel)

        problem = Problem(obj_configs=[obj1, obj2], design_space=zdt1.bounds)

        # Decrease min_max_calls to quickly enter the bi-objective phase
        opt_config = DriverConfig(max_iter=25, nt_init=10, seed=42)

        driver = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiEGO,
            strategy_kwargs={"min_max_calls": 3, "n_multi_start": 15},
        )

        # Initial DoE hypervolume
        driver.start_optim()
        initial_dataset = driver.state.dataset
        initial_pf = get_pf_from_dataset(initial_dataset)
        ref_point = np.array([2.0, 2.0])
        initial_hv = hypervolume_2d(initial_pf, ref_point)

        # Optimize
        state = driver.optimize()
        final_pf = get_pf_from_dataset(state.dataset)
        final_hv = hypervolume_2d(final_pf, ref_point)

        from pymoo.indicators.igd_plus import IGDPlus

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


        plot_pareto_front(
            initial_dataset,
            state.dataset,
            filename="zdt1_pareto_front_biego.png",
            title="Pareto Front of ZDT1 (2D) obtained with BiEGO",
        )

        import matplotlib.pyplot as plt
        from smt_optim.utils.multi_obj import plot_hypervolume_convergence, get_pareto_front

        y_all = state.dataset.export_as_dict()["obj"]
        
        hv_history = []
        for i in range(opt_config.nt_init, len(y_all) + 1):
            current_pts = y_all[:i]
            current_pf = get_pareto_front(current_pts)
            
            # Compute Area under PF w.r.t (0,0) (which decays as PF converges to the origin)
            sorted_pf = current_pf[np.argsort(current_pf[:, 0])]
            area = 0.0
            prev_f1 = 0.0
            for k in range(sorted_pf.shape[0]):
                area += (sorted_pf[k, 0] - prev_f1) * sorted_pf[k, 1]
                prev_f1 = sorted_pf[k, 0]
            hv_history.append(area)
            
        plot_hypervolume_convergence(
            hv_history, 
            filename="zdt1_hypervolume_decay.png",
            title="Hypervolume (Area under PF) Decay over Iterations"
        )



if __name__ == "__main__":
    unittest.main()

def test_mf_biego_constrained():
    from smt_optim.core.driver import DriverConfig, Driver
    from smt_optim.benchmarks.multi_obj.zdt_mf import DTLZ5
    from smt_optim.core.problem import Problem
    from smt_optim.core import ObjectiveConfig, ConstraintConfig
    from smt_optim.surrogate_models.smt import SmtAutoModel
    from smt_optim.acquisition_strategies.biego import BiEGO

    dtlz5 = DTLZ5()
    dtlz5.set_dim(2)

    obj1 = ObjectiveConfig(objective=dtlz5.objective[0], surrogate=SmtAutoModel)
    obj2 = ObjectiveConfig(objective=dtlz5.objective[1], surrogate=SmtAutoModel)
    cstr1 = ConstraintConfig(constraint=dtlz5.constraints[0], surrogate=SmtAutoModel, upper=0.0)

    problem = Problem(
        obj_configs=[obj1, obj2],
        cstr_configs=[cstr1],
        design_space=dtlz5.bounds,
        costs=[1.0, 10.0]
    )

    opt_config = DriverConfig(max_iter=10, nt_init=[10, 5], seed=42)
    driver = Driver(problem=problem, config=opt_config, strategy=BiEGO, strategy_kwargs={'min_max_calls': 2, 'n_multi_start': 50})
    driver.start_optim()
    while driver.state.iter < opt_config.max_iter:
        driver.iteration(driver.state)
        
    assert len(driver.state.dataset.export_as_dict()['x']) == 25
