import unittest
import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies.biego import BiEGO
from smt_optim.utils.multi_obj import get_pf_from_dataset, hypervolume_2d


def f1(x: np.ndarray) -> np.ndarray:
    ndim = x.ndim
    if ndim == 1:
        x = x.reshape(1, -1)
    value = x[:, 0] ** 2 + x[:, 1] ** 2
    if ndim == 1:
        return value.item()
    return value


def f2(x: np.ndarray) -> np.ndarray:
    ndim = x.ndim
    if ndim == 1:
        x = x.reshape(1, -1)
    value = (x[:, 0] - 1) ** 2 + (x[:, 1] - 1) ** 2
    if ndim == 1:
        return value.item()
    return value


def cstr(x: np.ndarray) -> np.ndarray:
    ndim = x.ndim
    if ndim == 1:
        x = x.reshape(1, -1)
    value = x[:, 0] ** 2 + x[:, 1] ** 2 - 4.0
    if ndim == 1:
        return value.item()
    return value


class TestBiEGO(unittest.TestCase):
    def test_biego_2d_1c(self):
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])

        obj_config1 = ObjectiveConfig(
            objective=[f1],
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            objective=[f2],
            surrogate=SmtAutoModel,
        )

        cstr_config = ConstraintConfig(
            constraint=[cstr],
            upper=0.0,
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1, obj_config2],
            cstr_configs=[cstr_config],
            design_space=bounds,
        )

        opt_config = DriverConfig(
            max_iter=3,
            seed=42,
        )

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiEGO,
        )

        state = optimizer.optimize()

        y_data = np.empty((len(state.dataset.samples), 2))
        for i, sample in enumerate(state.dataset.samples):
            y_data[i, 0] = sample.obj[0]
            y_data[i, 1] = sample.obj[1]

        self.assertGreater(len(y_data), 3)

    def test_biego_convergence(self):
        # A test to verify if the hypervolume strictly increases over several iterations
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])

        obj_config1 = ObjectiveConfig(
            objective=[f1],
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            objective=[f2],
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1, obj_config2],
            design_space=bounds,
        )

        opt_config = DriverConfig(
            max_iter=10,
            seed=42,
        )

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiEGO,
        )

        # Run optimization
        state = optimizer.optimize()

        # Get initial dataset
        num_iters = opt_config.max_iter
        num_total = len(state.scaled_dataset.export_as_dict()["obj"])
        num_init = num_total - num_iters

        # We construct the pareto front of the initial design points
        initial_objs = state.scaled_dataset.export_as_dict()["obj"][:num_init]
        # Quick pareto filtering for initial points
        is_efficient = np.ones(num_init, dtype=bool)
        for i, c in enumerate(initial_objs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    initial_objs[is_efficient] < c, axis=1
                )
                is_efficient[i] = True

        initial_pf = initial_objs[is_efficient]

        ref_point = np.array([10.0, 10.0])
        initial_hv = hypervolume_2d(initial_pf, ref_point)

        final_pf = get_pf_from_dataset(state.scaled_dataset)
        final_hv = hypervolume_2d(final_pf, ref_point)

        # Hypervolume of the Pareto front relative to [10, 10] should INCREASE
        # as we minimize and push the front towards (0,0) / (1,1)
        self.assertGreater(final_hv, initial_hv)


if __name__ == "__main__":
    unittest.main()
