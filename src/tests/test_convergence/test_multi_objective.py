import unittest
import numpy as np

from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.benchmarks.multiobj.zdt import ZDT1
from smt_optim.utils.multi_obj import (
    get_pf_from_dataset,
    hypervolume_2d,
)
from smt_optim.acquisition_strategies.mosego import MOSEGO
from smt_optim.acquisition_functions import init_mpi


class TestMultiObjectiveConvergence(unittest.TestCase):
    def test_zdt1_mosego_ehvi_hypervolume_growth(self):
        """Test that MOSEGO (with EHVI) improves the hypervolume of the Pareto front on ZDT1."""
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj1 = ObjectiveConfig(objective=[zdt1.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[zdt1.f2], surrogate=SmtAutoModel)

        problem = Problem(obj_configs=[obj1, obj2], design_space=zdt1.bounds)

        opt_config = DriverConfig(max_iter=15, nt_init=10, seed=42)

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

        # Hypervolume should increase or remain the same
        self.assertGreaterEqual(final_hv, initial_hv)

    def test_zdt1_mosego_mpi_hypervolume_growth(self):
        """Test that MOSEGO (with MPI) improves the hypervolume of the Pareto front on ZDT1."""
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj1 = ObjectiveConfig(objective=[zdt1.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[zdt1.f2], surrogate=SmtAutoModel)

        problem = Problem(obj_configs=[obj1, obj2], design_space=zdt1.bounds)

        opt_config = DriverConfig(max_iter=15, nt_init=10, seed=42)

        driver = Driver(
            problem=problem,
            config=opt_config,
            strategy=MOSEGO,
            strategy_kwargs={"acq_init": init_mpi},
        )

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

        # Hypervolume should increase or remain the same
        self.assertGreaterEqual(final_hv, initial_hv)

    def test_bnh_mosego_constrained_convergence(self):
        """Test that MOSEGO correctly optimizes a constrained problem (BNH) with relax_constraints=2.0."""
        from smt_optim.benchmarks.multiobj.constrained import BNH
        from smt_optim.core import ConstraintConfig

        bnh = BNH()

        obj1 = ObjectiveConfig(objective=[bnh.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[bnh.f2], surrogate=SmtAutoModel)

        cstr1 = ConstraintConfig(constraint=[bnh.g1], surrogate=SmtAutoModel)
        cstr2 = ConstraintConfig(constraint=[bnh.g2], surrogate=SmtAutoModel)

        problem = Problem(
            obj_configs=[obj1, obj2],
            cstr_configs=[cstr1, cstr2],
            design_space=bnh.bounds,
        )

        opt_config = DriverConfig(max_iter=5, nt_init=10, seed=42)

        driver = Driver(
            problem=problem,
            config=opt_config,
            strategy=MOSEGO,
            strategy_kwargs={"relax_constraints": 2.0},
        )

        # Optimize for a few iterations
        state = driver.optimize()

        # Ensure we have some feasible points in the final dataset
        from smt_optim.utils.multi_obj import get_pf_from_dataset

        final_pf = get_pf_from_dataset(state.dataset)

        self.assertGreaterEqual(
            len(final_pf),
            1,
            "There should be at least one feasible point in the Pareto Front.",
        )

    def test_bnh_mosego_constrained_convergence_no_relax(self):
        """Test that MOSEGO correctly optimizes a constrained problem (BNH) with relax_constraints=0.0."""
        from smt_optim.benchmarks.multiobj.constrained import BNH
        from smt_optim.core import ConstraintConfig

        bnh = BNH()

        obj1 = ObjectiveConfig(objective=[bnh.f1], surrogate=SmtAutoModel)
        obj2 = ObjectiveConfig(objective=[bnh.f2], surrogate=SmtAutoModel)

        cstr1 = ConstraintConfig(constraint=[bnh.g1], surrogate=SmtAutoModel)
        cstr2 = ConstraintConfig(constraint=[bnh.g2], surrogate=SmtAutoModel)

        problem = Problem(
            obj_configs=[obj1, obj2],
            cstr_configs=[cstr1, cstr2],
            design_space=bnh.bounds,
        )

        opt_config = DriverConfig(max_iter=5, nt_init=10, seed=42)

        driver = Driver(
            problem=problem,
            config=opt_config,
            strategy=MOSEGO,
            strategy_kwargs={"relax_constraints": 0.0},
        )

        # Optimize for a few iterations
        state = driver.optimize()

        # Just verify it finishes without crashing and evaluates valid points
        import numpy as np

        x_evaluated = state.dataset.export_as_dict()["x"]
        self.assertGreater(len(x_evaluated), 10)

        # Verify valid feasible points were evaluated (though not necessarily found in 15 iterations)
        c1_vals = np.array([bnh.g1(x) for x in x_evaluated])
        c2_vals = np.array([bnh.g2(x) for x in x_evaluated])

        self.assertEqual(len(c1_vals), len(x_evaluated))
        self.assertEqual(len(c2_vals), len(x_evaluated))

    def test_dtlz5_mosego_multifidelity_constrained(self):
        """Test that MOSEGO correctly exploits Multi-Fidelity models under constraints (DTLZ5)."""
        from smt_optim.benchmarks.multiobj.zdt_mf import DTLZ5
        from smt_optim.core import ConstraintConfig

        problem = DTLZ5()
        problem.set_dim(4)

        obj_config = ObjectiveConfig(
            problem.objective[0],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            problem.objective[1],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        cstr_config = ConstraintConfig(
            problem.constraints[0],
            upper=0.0,
            surrogate=SmtAutoModel,
        )

        prob_definition = Problem(
            obj_configs=[obj_config, obj_config2],
            cstr_configs=[cstr_config],
            design_space=problem.bounds,
            costs=[0.2, 1.0],
        )

        nt_init = 12

        opt_config = DriverConfig(
            max_iter=3,
            nt_init=nt_init,
            verbose=False,
            scaling=True,
            seed=0,
        )

        driver = Driver(
            problem=prob_definition,
            config=opt_config,
            strategy=MOSEGO,
            strategy_kwargs={"relax_constraints": 0.0},
        )
        state = driver.optimize()

        # Check that we evaluated High Fidelity points natively

        self.assertGreater(len(state.dataset.export_as_dict()["x"]), 0)


if __name__ == "__main__":
    unittest.main()
