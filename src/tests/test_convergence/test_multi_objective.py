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

        opt_config = DriverConfig(max_iter=10, nt_init=5, seed=42)

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
        opt_config = DriverConfig(max_iter=25, nt_init=5, seed=42)

        driver = Driver(
            problem=problem, 
            config=opt_config, 
            strategy=BiEGO,
            strategy_kwargs={"min_max_calls": 2}
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
        # Step by step plotting
        r_history = driver.strategy.r_history
        # The bi-objective phase starts after 2*min_max_calls = 4 iterations.
        # So infills from index 4 onwards are bi-objective infills.
        # But wait! If min_max_calls=2, then iterations 1,2 are min f1, 3,4 are min f2.
        # Iteration 5 is bi-objective, producing infill 5.
        
        y_all = state.dataset.export_as_dict()["obj"]
        y_init = y_all[:opt_config.nt_init]
        y_infills = y_all[opt_config.nt_init:]
        
        # We start plotting from the first bi-objective iteration
        # which corresponds to the first element in r_history.
        # Wait, not every step has a new r_history element if r is None (empty pareto front).
        # But for ZDT1 with 5 initial points, PF is not empty.
        
        r_idx = 0
        for i in range(len(y_infills)):
            step_dataset = state.dataset.__class__()
            for j in range(opt_config.nt_init + i + 1):
                step_dataset.add(state.dataset.samples[j])
                
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x1, 1 - np.sqrt(x1), "k-", label="Ref PF", linewidth=2)
            ax.plot(y_init[:, 0], y_init[:, 1], "go", label="Initial DoE", alpha=0.7)
            
            # Plot previous infills
            if i > 0:
                ax.plot(y_infills[:i, 0], y_infills[:i, 1], "bo", label="Previous Infills", alpha=0.7)
                
            # Plot the new infill
            ax.plot(y_infills[i, 0], y_infills[i, 1], "r*", markersize=12, label="New Infill")
            
            # Plot current Nadir if in bi-objective phase
            # We assume that after 4 single-objective infills, we are in bi-objective.
            if i >= 4 and r_idx < len(r_history):
                r = r_history[r_idx]
                ax.plot(r[0], r[1], "mX", markersize=10, label="Adaptive Nadir (r)")
                ax.axhline(r[1], color="m", linestyle="--", alpha=0.5)
                ax.axvline(r[0], color="m", linestyle="--", alpha=0.5)
                r_idx += 1
            
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
            ax.set_title(f"BiEGO Step {i+1}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(f"zdt1_biego_step_{i+1}.png", dpi=150, bbox_inches="tight")
            plt.close()



if __name__ == "__main__":
    unittest.main()
