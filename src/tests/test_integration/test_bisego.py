import unittest

import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, DriverConfig, Driver

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import BiSEGO


from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver

from smt_optim.benchmarks.registry import get_problem
from pymoo.indicators.hv import HV


def dominates(p,q):
    # Returns True if point p strictly dominates point q, else returns False
    return (p[0]<q[0] and p[1]<=q[1]) or (p[0]<=q[0] and p[1]<q[1])

def get_Pareto_front(Y):
    # Given a list Y of points in the objective space, returns the list of non-dominated points.
    t=len(Y)
    front=[]
    for i in range(t):
        if all([not dominates(q,Y[i]) for q in Y]):
            front.append(Y[i])
    front.sort()
    return front


class TestOptimization(unittest.TestCase):
    def test_bisego_composite(self):
        bproblem = get_problem("ZDT1")

        num_obj=bproblem.num_obj
        num_cstr=bproblem.num_cstr
        bounds=bproblem.bounds
        objective=bproblem.objective

        assert(num_obj==2)
        assert(num_cstr==0)

        f1=objective[0]
        f2=objective[1]


        obj_config1 = ObjectiveConfig(
            objective=[f1],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            objective=[f2],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1,obj_config2],
            design_space=bounds,
            costs=[1,1]
        )

        opt_config = DriverConfig(
            max_iter=25,
            seed=43,
            nt_init=10,
        )

        strategy_kwargs = {
            "n_multi_start":10,
            "n_accuracy":100,
            "so_formulation":"Product",
            "single_objective_max_calls":5,
            "init_calls":5,
            "naive":False,
            "verbose":False,
        }

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiSEGO,
            strategy_kwargs=strategy_kwargs,
        )

        state = optimizer.optimize()

        y_data = []
        for i, sample in enumerate(state.dataset.samples):
            y_data.append((sample.obj[0],sample.obj[1]))
        
        ref_point=np.array([1,1])
        pareto_points_composite=get_Pareto_front(y_data)

        HV_max=HV(ref_point=ref_point).do(np.array(pareto_points_composite))

        self.assertGreaterEqual(HV_max,0.4)
        # Theoretical maximum is 0.66

    def test_bisego_composite_normalized(self):
        bproblem = get_problem("ZDT1")

        num_obj=bproblem.num_obj
        num_cstr=bproblem.num_cstr
        bounds=bproblem.bounds
        objective=bproblem.objective

        assert(num_obj==2)
        assert(num_cstr==0)

        f1=objective[0]
        f2=objective[1]


        obj_config1 = ObjectiveConfig(
            objective=[f1],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            objective=[f2],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1,obj_config2],
            design_space=bounds,
            costs=[1,1]
        )

        opt_config = DriverConfig(
            max_iter=35,
            seed=43,
            nt_init=10,
        )

        strategy_kwargs = {
            "n_multi_start":10,
            "so_formulation":"Normalized",
            "single_objective_max_calls":5,
            "init_calls":5,
            "naive":False,
            "verbose":False,
        }

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiSEGO,
            strategy_kwargs=strategy_kwargs,
        )

        state = optimizer.optimize()

        y_data = []
        for i, sample in enumerate(state.dataset.samples):
            y_data.append((sample.obj[0],sample.obj[1]))
        
        ref_point=np.array([1,1])
        pareto_points_composite=get_Pareto_front(y_data)

        HV_max=HV(ref_point=ref_point).do(np.array(pareto_points_composite))

        self.assertGreaterEqual(HV_max,0.4)
        # Theoretical maximum is 0.66

    def test_bisego_naive(self):
        bproblem = get_problem("ZDT1")

        num_obj=bproblem.num_obj
        num_cstr=bproblem.num_cstr
        bounds=bproblem.bounds
        objective=bproblem.objective

        assert(num_obj==2)
        assert(num_cstr==0)

        f1=objective[0]
        f2=objective[1]


        obj_config1 = ObjectiveConfig(
            objective=[f1],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        obj_config2 = ObjectiveConfig(
            objective=[f2],
            type="minimize",
            surrogate=SmtAutoModel,
        )

        problem = Problem(
            obj_configs=[obj_config1,obj_config2],
            design_space=bounds,
            costs=[1,1]
        )

        opt_config = DriverConfig(
            max_iter=35,
            seed=43,
            nt_init=10,
        )

        strategy_kwargs = {
            "n_multi_start":10,
            "n_accuracy":100,
            "so_formulation":"Normalized",
            "single_objective_max_calls":5,
            "init_calls":5,
            "naive":True,
            "verbose":False,
        }

        optimizer = Driver(
            problem=problem,
            config=opt_config,
            strategy=BiSEGO,
            strategy_kwargs=strategy_kwargs,
        )

        state = optimizer.optimize()

        y_data = []
        for i, sample in enumerate(state.dataset.samples):
            y_data.append((sample.obj[0],sample.obj[1]))
        
        ref_point=np.array([1,1])
        pareto_points_composite=get_Pareto_front(y_data)

        HV_max=HV(ref_point=ref_point).do(np.array(pareto_points_composite))

        self.assertGreaterEqual(HV_max,0.4)
        # Theoretical maximum is 0.66



if __name__ == "__main__":
    unittest.main()
