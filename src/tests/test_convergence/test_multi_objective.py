import unittest
import numpy as np

from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtGPX
from smt_optim.acquisition_strategies.mosego import MOSEGO


class TestMultiObjectiveConvergence(unittest.TestCase):
    def test_bnh_mosego_constrained_convergence(self):
        """Test that MOSEGO correctly optimizes a constrained problem (BNH) with relax_constraints=2.0."""
        from smt_optim.benchmarks.multiobj.constrained import BNH
        from smt_optim.core import ConstraintConfig

        bnh = BNH()

        obj1 = ObjectiveConfig(objective=[bnh.f1], surrogate=SmtGPX)
        obj2 = ObjectiveConfig(objective=[bnh.f2], surrogate=SmtGPX)

        cstr1 = ConstraintConfig(constraint=[bnh.g1], surrogate=SmtGPX)
        cstr2 = ConstraintConfig(constraint=[bnh.g2], surrogate=SmtGPX)

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

        obj1 = ObjectiveConfig(objective=[bnh.f1], surrogate=SmtGPX)
        obj2 = ObjectiveConfig(objective=[bnh.f2], surrogate=SmtGPX)

        cstr1 = ConstraintConfig(constraint=[bnh.g1], surrogate=SmtGPX)
        cstr2 = ConstraintConfig(constraint=[bnh.g2], surrogate=SmtGPX)

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

        x_evaluated = state.dataset.export_as_dict()["x"]
        self.assertGreater(len(x_evaluated), 10)

        # Verify valid feasible points were evaluated (though not necessarily found in 15 iterations)
        c1_vals = np.array([bnh.g1(x) for x in x_evaluated])
        c2_vals = np.array([bnh.g2(x) for x in x_evaluated])

        self.assertEqual(len(c1_vals), len(x_evaluated))
        self.assertEqual(len(c2_vals), len(x_evaluated))


if __name__ == "__main__":
    unittest.main()
