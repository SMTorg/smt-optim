import unittest
import numpy as np
from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.benchmarks.multiobj.zdt import ZDT1
from smt_optim.utils.multi_obj import get_pf_from_dataset, hypervolume_2d
from smt_optim.acquisition_strategies.mosego import MOSEGO


class TestZDT1HVIntegration(unittest.TestCase):
    def test_zdt1_mosego_hv(self):
        zdt1 = ZDT1()
        zdt1.set_dim(2)

        obj_config1 = ObjectiveConfig(
            objective=[zdt1.f1],
            surrogate=SmtAutoModel,
        )
        obj_config2 = ObjectiveConfig(
            objective=[zdt1.f2],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1, obj_config2],
            design_space=zdt1.bounds,
        )

        opt_config = DriverConfig(
            max_iter=3,
            seed=42,
        )

        ref_point = np.array([2.0, 2.0])

        driver_mosego = Driver(problem=problem, config=opt_config, strategy=MOSEGO)
        state_mosego = driver_mosego.optimize()
        pf_mosego = get_pf_from_dataset(state_mosego.scaled_dataset)
        hv_mosego = hypervolume_2d(pf_mosego, ref_point)

        self.assertGreater(len(state_mosego.dataset.samples), 0)
        self.assertGreater(hv_mosego, 0)


if __name__ == "__main__":
    unittest.main()
