import unittest

import numpy as np

from smt_optim.core import (
    Driver,
    ObjectiveConfig,
    ConstraintConfig,
    DriverConfig,
    Problem,
)

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import MOSEGO
from smt_optim.utils.multi_obj import get_pf_from_dataset


class TestParetoFront(unittest.TestCase):
    def test_get_pf_from_dataset_constrained(self):

        from smt_optim.benchmarks.multiobj.zdt_mf import DTLZ5

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

        # ------- Unconstrained problem -------
        prob_definition = Problem(
            obj_configs=[obj_config, obj_config2],
            # cstr_configs=[cstr_config],
            design_space=problem.bounds,  # problem bounds
            costs=[0.5, 1.0],
        )

        nt_init = 200

        opt_config = DriverConfig(
            max_iter=0,
            nt_init=nt_init,
            verbose=False,
            scaling=True,
            seed=0,
        )

        driver = Driver(prob_definition, opt_config, MOSEGO)
        state = driver.optimize()

        pareto_front_uncstr = get_pf_from_dataset(state.dataset)

        max_o1_uncstr = pareto_front_uncstr[:, 0].max()

        # ------- Constrained problem -------
        prob_definition = Problem(
            obj_configs=[obj_config, obj_config2],
            cstr_configs=[cstr_config],
            design_space=problem.bounds,  # problem bounds
            costs=[0.5, 1.0],
        )

        driver = Driver(prob_definition, opt_config, MOSEGO)
        state = driver.optimize()

        pareto_front_cstr = get_pf_from_dataset(state.dataset)
        max_o1_cstr = pareto_front_cstr[:, 0].max()

        # the constraint should reduce the maximum feasible value of the first objective (<= 0.5, vs <= 1.0)
        self.assertLessEqual(max_o1_cstr, max_o1_uncstr)

        pareto_front_relax = get_pf_from_dataset(state.dataset, ctol=np.inf)
        max_o1_relax = pareto_front_relax[:, 0].max()

        # relaxing the constraint should give the same PF as the unconstrained problem PF
        self.assertEqual(max_o1_relax, max_o1_uncstr)
        self.assertEqual(pareto_front_relax.shape[0], pareto_front_uncstr.shape[0])

        pareto_dict = get_pf_from_dataset(state.dataset, return_dict=True)

        # the items in the dictionary should have the same number of elements as the objective vector
        self.assertEqual(pareto_front_cstr.shape[0], pareto_dict["obj"].shape[0])


if __name__ == "__main__":
    unittest.main()
