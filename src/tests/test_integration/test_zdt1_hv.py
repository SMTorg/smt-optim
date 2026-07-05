import numpy as np
from smt_optim.core import Problem, ObjectiveConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.benchmarks.multi_obj.zdt import ZDT1
from smt_optim.utils.multi_obj import get_pf_from_dataset, hypervolume_2d

from smt_optim.acquisition_strategies.biego import BiEGO
from smt_optim.acquisition_strategies.mosego import MOSEGO


def main():
    print("Setting up ZDT1 benchmark (dim=2)...")

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
        max_iter=15,
        seed=42,
    )

    ref_point = np.array([2.0, 2.0])

    print("\n--- Testing Classic MOSEGO ---")
    driver_mosego = Driver(problem=problem, config=opt_config, strategy=MOSEGO)
    state_mosego = driver_mosego.optimize()
    pf_mosego = get_pf_from_dataset(state_mosego.scaled_dataset)
    hv_mosego = hypervolume_2d(pf_mosego, ref_point)
    print(f"MOSEGO Finished. Total evaluations: {len(state_mosego.dataset.samples)}")
    print(f"MOSEGO Hypervolume (ref [2, 2]): {hv_mosego:.4f}")

    print("\n--- Testing BiEGO ---")
    driver_biego = Driver(problem=problem, config=opt_config, strategy=BiEGO)
    state_biego = driver_biego.optimize()
    pf_biego = get_pf_from_dataset(state_biego.scaled_dataset)
    hv_biego = hypervolume_2d(pf_biego, ref_point)
    print(f"BiEGO Finished. Total evaluations: {len(state_biego.dataset.samples)}")
    print(f"BiEGO Hypervolume (ref [2, 2]): {hv_biego:.4f}")


if __name__ == "__main__":
    main()
