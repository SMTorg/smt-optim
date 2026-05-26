import unittest

import numpy as np

from smt_optim.core import (
    Driver,
    ObjectiveConfig,
    ConstraintConfig,
    DriverConfig,
    Problem,
)

from smt_optim.surrogate_models.smt import SmtGPX, SmtAutoModel

from smt_optim.acquisition_strategies import MOSEGO
from smt_optim.acquisition_functions import init_mpi, init_ehvi_2o
from smt_optim.utils.multi_obj import get_pf_from_dataset


from smt_optim.benchmarks.base import PymooWrapper
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class TestMOSEGO(unittest.TestCase):
    def test_no_constraint(self):
        """
        Test convergence of MO-SEGO with EHVI (2o) acquisition function.

        Returns
        -------

        """
        from smt_optim.benchmarks.multiobj.zdt import ZDT1

        x_doe = np.array(
            [
                [0.56265405, 0.98701448],
                [0.72132716, 0.05395734],
                [0.20819470, 0.20330553],
                [0.12739234, 0.74589931],
                [0.9087250, 0.58255112],
            ]
        )

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        problem = ZDT1()
        problem.set_dim(2)

        # initialize the first objective configuration
        obj1_config = ObjectiveConfig(
            [problem.f1],
            type="minimize",
            surrogate=SmtGPX,
        )

        # initialize the second objective configuration
        obj2_config = ObjectiveConfig(
            [problem.f2],
            type="minimize",
            surrogate=SmtGPX,
        )

        # initialize the problem configuration
        prob_definition = Problem(
            obj_configs=[obj1_config, obj2_config],
            design_space=problem.bounds,  # problem bounds
        )

        # initialize the driver
        opt_config = DriverConfig(
            max_iter=20,
            xt_init=[x_doe],
            verbose=False,
            scaling=True,
            seed=0,
        )

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_init": init_ehvi_2o, "n_start": 20},
        )

        # starts the optimization process
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset)

        pymoo_prob = PymooWrapper(problem)
        algorithm = NSGA2(pop_size=20, seed=0)
        res = minimize(pymoo_prob, algorithm, ("n_gen", 100), seed=0)

        hv_bo = hv(obj_par)
        hv_nsga2 = hv(res.F)
        rel_error = abs(hv_bo - hv_nsga2) / hv_nsga2

        # fig, ax = plt.subplots()
        # ax.scatter(res.F[:, 0], res.F[:, 1], color="C7", alpha=0.5)
        # ax.scatter(obj_par[:, 0], obj_par[:, 1], color="C0")
        # plt.show()

        self.assertLess(rel_error, 0.1)

    def test_mpi(self):
        """
        Test the HV improvement between the initial and final DOE with the MPI acquisition function.

        Returns
        -------

        """
        from smt_optim.benchmarks.multiobj.zdt import ZDT1

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        x_doe = np.array(
            [
                [0.56265405, 0.98701448],
                [0.72132716, 0.05395734],
                [0.20819470, 0.20330553],
                [0.12739234, 0.74589931],
                [0.9087250, 0.58255112],
            ]
        )

        problem = ZDT1()
        problem.set_dim(2)

        # initialize the first objective configuration
        obj1_config = ObjectiveConfig(
            [problem.f1],
            type="minimize",
            surrogate=SmtGPX,
        )

        # initialize the second objective configuration
        obj2_config = ObjectiveConfig(
            [problem.f2],
            type="minimize",
            surrogate=SmtGPX,
        )

        # initialize the problem configuration
        prob_definition = Problem(
            obj_configs=[obj1_config, obj2_config],
            design_space=problem.bounds,  # problem bounds
        )

        # initialize the driver
        opt_config = DriverConfig(
            max_iter=20,
            xt_init=[x_doe],
            verbose=False,
            scaling=False,
            seed=0,
        )

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_init": init_mpi, "n_start": 20},
        )

        # starts the optimization process
        driver.config.max_iter = 0
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset)

        hv_init = hv(obj_par)

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_init": init_mpi, "n_start": 20},
        )

        driver.config.max_iter = 20
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset)

        hv_final = hv(obj_par)

        self.assertGreater(hv_final, hv_init)

    def test_with_constraints(self):
        """
        Test the HV improvement between the initial and final DOE when dealing with constrained problem.

        Test that the HV is greater when the constraint tolerance is relaxed.

        Returns
        -------

        """
        from smt_optim.benchmarks.multiobj.constrained import TNK

        x_doe = np.array(
            [
                [1.04979075, 1.29826814],
                [0.94615388, 0.70670154],
                [0.43618902, 1.15416911],
                [0.57972416, 0.75932125],
                [0.66473842, 0.07623957],
                [0.51430532, 0.45251892],
                [0.07807301, 0.53817417],
                [1.49505326, 0.27971224],
                [1.38643765, 1.25411163],
                [0.87529683, 0.37520539],
                [0.69747839, 1.41671159],
                [0.75212398, 1.04856265],
                [1.10141565, 0.020234],
                [1.17663350, 0.37013043],
                [0.27049768, 0.87353921],
                [0.34077187, 1.45183464],
                [0.04777213, 1.09878445],
                [1.21013224, 0.92877582],
                [0.21099527, 0.21845667],
                [1.31440157, 0.64060959],
            ]
        )

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        problem = TNK()

        problem.bounds = np.array([[0.0, 1.5]] * 2)

        # initialize the first objective configuration
        obj1_config = ObjectiveConfig(
            [problem.f1],
            type="minimize",
            surrogate=SmtGPX,
        )

        # initialize the second objective configuration
        obj2_config = ObjectiveConfig(
            [problem.f2],
            type="minimize",
            surrogate=SmtGPX,
        )

        cstr1_config = ConstraintConfig(
            [problem.constraints[0]],
            upper=0.0,
            surrogate=SmtGPX,
        )

        cstr2_config = ConstraintConfig(
            [problem.constraints[1]],
            upper=0.0,
            surrogate=SmtGPX,
        )

        # initialize the problem configuration
        prob_definition = Problem(
            obj_configs=[obj1_config, obj2_config],
            cstr_configs=[cstr1_config, cstr2_config],
            design_space=problem.bounds,  # problem bounds
        )

        # initialize the driver
        opt_config = DriverConfig(
            max_iter=10,
            xt_init=[x_doe],
            verbose=False,
            scaling=True,
            seed=0,
        )

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_init": init_ehvi_2o, "n_start": 40},
        )

        driver.config.max_iter = 0
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset, ctol=1e-4)

        hv_init = hv(obj_par)

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={"acq_init": init_ehvi_2o, "n_start": 40},
        )

        driver.config.max_iter = 10
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset, ctol=1e-4)
        hv_final = hv(obj_par)

        # Hypervolume is improved between after the optimization
        self.assertLess(hv_init, hv_final)

        obj_par = get_pf_from_dataset(state.dataset, ctol=1.0)
        hv_relax = hv(obj_par)

        # Hypervolume increases when the constraint tolerance is relaxed
        self.assertGreater(hv_relax, hv_final)

    def test_multi_fidelity(self):
        """
        Test the HV improvement between the initial and final DOE when dealing with a multi-fidelity problem.

        Returns
        -------

        """
        from smt_optim.benchmarks.multiobj.zdt_mf import DTLZ5

        x_doe_0 = np.array(
            [
                [7.70227592e-01, 9.47729525e-01, 7.11532110e-01, 8.47105656e-01],
                [1.92870214e-01, 5.94513718e-01, 2.04867620e-03, 5.16895561e-01],
                [5.28576492e-01, 6.52928402e-01, 1.40792678e-01, 8.64748280e-02],
                [4.74291768e-01, 4.36074417e-01, 9.97447184e-01, 9.21261431e-01],
                [9.81010673e-01, 2.27073061e-01, 2.14985595e-01, 4.15512094e-01],
                [8.68255508e-01, 8.55274764e-01, 5.29715002e-01, 2.82359476e-01],
                [2.43158946e-01, 9.99754825e-01, 3.84422337e-01, 3.69446071e-01],
                [2.51415984e-01, 8.33599744e-01, 3.49860497e-01, 8.96357728e-01],
                [9.06635120e-02, 2.56214164e-01, 5.61357880e-01, 7.79016619e-01],
                [9.22018858e-01, 3.19183878e-01, 9.24994791e-01, 2.21134361e-01],
                [2.68324708e-01, 4.77532480e-01, 3.93271623e-01, 9.23732105e-01],
                [8.81296120e-01, 3.51947182e-01, 4.02062816e-01, 4.94001399e-02],
                [6.21623315e-01, 6.20822209e-01, 9.73868477e-01, 2.35135764e-01],
                [1.42352728e-01, 7.44826282e-01, 2.42667658e-01, 1.70636758e-01],
                [3.78844440e-01, 9.96624414e-01, 6.76233054e-01, 6.24203108e-01],
                [4.32413591e-01, 2.82084943e-02, 1.79507989e-01, 4.35206737e-01],
                [9.85089172e-01, 5.39713946e-01, 5.04836494e-01, 8.47930435e-01],
                [7.18125323e-01, 8.17123266e-01, 6.23688534e-04, 5.44788522e-01],
                [8.50988441e-02, 2.42703459e-01, 7.86928574e-01, 3.25967468e-01],
                [5.99072919e-01, 1.44625064e-01, 8.71835263e-01, 7.88649008e-01],
            ]
        )

        x_doe_1 = np.array(
            [
                [2.68324708e-01, 4.77532480e-01, 3.93271623e-01, 9.23732105e-01],
                [8.81296120e-01, 3.51947182e-01, 4.02062816e-01, 4.94001399e-02],
                [6.21623315e-01, 6.20822209e-01, 9.73868477e-01, 2.35135764e-01],
                [1.42352728e-01, 7.44826282e-01, 2.42667658e-01, 1.70636758e-01],
                [3.78844440e-01, 9.96624414e-01, 6.76233054e-01, 6.24203108e-01],
                [4.32413591e-01, 2.82084943e-02, 1.79507989e-01, 4.35206737e-01],
                [9.85089172e-01, 5.39713946e-01, 5.04836494e-01, 8.47930435e-01],
                [7.18125323e-01, 8.17123266e-01, 6.23688534e-04, 5.44788522e-01],
                [8.50988441e-02, 2.42703459e-01, 7.86928574e-01, 3.25967468e-01],
                [5.99072919e-01, 1.44625064e-01, 8.71835263e-01, 7.88649008e-01],
            ]
        )

        ref_point = np.array([1.0, 1.0])
        hv = HV(ref_point=ref_point)

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
            design_space=problem.bounds,  # problem bounds
            costs=[0.5, 1.0],
        )

        opt_config = DriverConfig(
            max_iter=10,
            xt_init=[x_doe_0, x_doe_1],
            verbose=False,
            scaling=True,
            seed=0,
        )

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={
                "acq_init": init_ehvi_2o,
                "n_start": 20,
                "sp_method": "SLSQP",
            },
        )

        driver.config.max_iter = 0
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset, ctol=1e-4)
        hv_init = hv(obj_par)

        driver = Driver(
            prob_definition,
            opt_config,
            MOSEGO,
            strategy_kwargs={
                "acq_init": init_ehvi_2o,
                "n_start": 20,
                "sp_method": "SLSQP",
            },
        )

        driver.config.max_iter = 10
        state = driver.optimize()

        obj_par = get_pf_from_dataset(state.dataset, ctol=1e-4)
        hv_final = hv(obj_par)

        self.assertGreater(hv_final, hv_init)


if __name__ == "__main__":
    unittest.main()
