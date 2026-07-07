from smt_optim.acquisition_strategies.bisego import BiEGO

import numpy as np

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver
from smt_optim.acquisition_strategies import MFSEGO

from smt_optim.benchmarks.registry import list_problems

import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from smt_optim.bisego_utils.naivebisego import NaiveBiEGO, InjectData, SimpleEGO
from smt.sampling_methods import LHS
import smt.design_space as ds

from datetime import datetime
import pickle

L=list_problems(num_obj=[2,2],tags=["zdt"])

surrogate=SmtAutoModel

a=1 # Set to 1 for testing, set to 3 for running Benchmark

# Parameters

n_accuracy=100 # Precision on composite acquisition function
seed=420 # Seed for generating the initial DoE
budget_factor=20 # Budget: budget_factor * dim
init_factor=2 # Initial DoE size: init_factor * dim + 1
min_factor=2 # Initial calls to determine min(f1): min_factor * dim + 1 (same for min(f2))
max_so_iter_factor=2 # Max calls to a single-objective subproblem: max_so_iter_factor * dim + 1
soformulation_naive="Normalized"
soformulation_composite="Product"
multi_start_factor=10 # Number of multistart calls for acquisition function optimization: multi_start_factor * dim
test_number=0

parameters={
    "n_accuracy":n_accuracy,
    "seed":seed,
    "budget_factor":budget_factor,
    "init_factor":init_factor,
    "min_factor":min_factor,
    "max_so_iter_factor":max_so_iter_factor,
    "soformulation_composite":soformulation_composite,
    "soformulation_naive":soformulation_naive,
    "multi_start_factor":multi_start_factor,
    "test_number":test_number,
}

def get_DoE(state):
    Y=state.dataset.export_as_dict()["obj"]
    D=state.dataset.export_as_dict()["x"]
    return (D,Y)

def Dominates(p,q):
    #Returns True if point p strictly dominates point q, else returns False
    return (p[0]<q[0] and p[1]<=q[1]) or (p[0]<=q[0] and p[1]<q[1])

def ParetoFront(D,Y):
    #Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    t=len(D)
    front=[]
    for i in range(t):
        if all([not Dominates(q,Y[i]) for q in Y]):
            front.append((Y[i][0],i))
    front.sort()
    return [p[1] for p in front]

def run_all_benchmarks(bproblem):
    #Initialization
    name=bproblem.name
    print("Running all benchmarks on problem",name)

    num_dim=bproblem.num_dim
    bounds=bproblem.bounds
    objective=bproblem.objective

    f1=objective[0]
    f2=objective[1]

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1

    float_vars = []
    for idx in range(bounds.shape[0]):
        float_vars.append(
            ds.FloatVariable(bounds[idx, 0], bounds[idx, 1])
        )
    design_space = ds.DesignSpace(float_vars)

    F=lambda x: (f1(x),f2(x))
    #init DoE using LHS with given seed
    sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(),
                              criterion="ese",
                              seed=seed, )
    doe = sampler(n_init)
    D=[x for x in doe]
    Y=[F(x) for x in D]
    Ni0=min_factor*num_dim+1
    Ng=max_budget-n_init-Ni0*2
    Ni=max_so_iter_factor*num_dim+1
    soformulation=soformulation_naive
    n_multistart=multi_start_factor*num_dim
    initial_budget=Ni0*2+n_init

    #run min f1 and min f2
    #Initialization
    t=len(D)
    W=[0]*t
    n_eval=0

    def F_eval(x):
        #This function is a substitute for F that automatically updates the global DoE and number of evals when called
        nonlocal n_eval,D,Y,W
        y=F(x)
        D.append(x)
        Y.append(y)
        W.append(0)
        n_eval+=1
        return y
    
    #Coordinate applications of f
    fa=lambda x:F_eval(x)[0]
    fb=lambda x:F_eval(x)[1]

    #Run EGO on the problems min(f1(x)) and min(f2(x))
    print("Min(f1) phase")
    state=SimpleEGO(fa,bounds,D,[y[0] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart})
    print("Min(f2) phase")
    state=SimpleEGO(fb,bounds,D,[y[1] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart})
    
    X=ParetoFront(D,Y)

    #D and Y are initialized

    #biego benchmark
    state_biego = run_benchmark(bproblem,D[:initial_budget],Y[:initial_budget])

    #naive benchmark
    pareto_points_naive,D_naive,Y_naive = NaiveBiEGO(F,D[:initial_budget],Y[:initial_budget],Ng,Ni,0,bounds,n_multistart=multi_start_factor*num_dim,soformulation=soformulation)
    
    #pymoo_benchmark
    X2,F2 = run_benchmark_pymoo_with_budget(bproblem,D[:initial_budget],Y[:initial_budget])

    #True Front
    X,F = run_benchmark_pymoo(bproblem)

    return state_biego,pareto_points_naive,D_naive,Y_naive,X,F,X2,F2



def run_benchmark(bproblem,D,Y):
    name=bproblem.name
    print("Running BiEGO on benchmark problem",name)
    num_dim=bproblem.num_dim
    num_obj=bproblem.num_obj
    num_cstr=bproblem.num_cstr
    bounds=bproblem.bounds
    objective=bproblem.objective

    assert(num_obj==2)
    assert(num_cstr==0)

    f1=objective[0]
    f2=objective[1]

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1
    n_min=min_factor*num_dim+1
    n_so=max_so_iter_factor*num_dim+1

    obj_config1 = ObjectiveConfig(
        [f1],
        type="minimize",
        surrogate=surrogate,
    )

    obj_config2 = ObjectiveConfig(
        [f2],
        type="minimize",
        surrogate=surrogate,
    )

    prob_definition = Problem(
        obj_configs=[obj_config1,obj_config2],
        design_space=bounds,            # problem bounds
        costs=[1,1]
    )


    opt_config = DriverConfig(
        max_iter = max_budget - n_init-2*n_min,
        max_budget = max_budget,
        nt_init = n_init+2*n_min,
        verbose = True,
        scaling = True,
        seed=seed,
    )

    strategy_kwargs = {
        "n_multi_start":multi_start_factor*num_dim,
        "n_init":0,
        "n_accuracy":n_accuracy,
        "so_formulation":soformulation_composite,
        "single_objective_max_calls":n_so,
        "min_max_calls":0
    }

    xt_init = np.vstack([np.atleast_1d(x) for x in D])
    yt_init = np.vstack([np.atleast_1d(y) for y in Y])

    driver = Driver(prob_definition, opt_config, strategy=BiEGO, strategy_kwargs=strategy_kwargs)

    InjectData(driver.state, xt_init, yt_init)

    state = driver.optimize()
    return state

def run_benchmark_naive(bproblem):
    name=bproblem.name
    print("Running naive BiEGO on benchmark problem",name)
    num_dim=bproblem.num_dim
    bounds=bproblem.bounds
    objective=bproblem.objective

    f1=objective[0]
    f2=objective[1]

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1

    float_vars = []
    for idx in range(bounds.shape[0]):
        float_vars.append(
            ds.FloatVariable(bounds[idx, 0], bounds[idx, 1])
        )
    design_space = ds.DesignSpace(float_vars)

    F=lambda x: (f1(x),f2(x))
    sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(),
                              criterion="ese",
                              seed=seed, )
    doe = sampler(n_init)
    D=[x for x in doe]
    Y=[F(x) for x in D]
    Ng=max_budget-n_init
    Ni=max_so_iter_factor*num_dim+1
    Ni0=min_factor*num_dim+1
    soformulation=soformulation_naive

    return NaiveBiEGO(F,D,Y,Ng,Ni,Ni0,bounds,n_multistart=multi_start_factor*num_dim,soformulation=soformulation)

class PymooProblem(ElementwiseProblem):

    def __init__(self,bprob):
        self.prob=bprob
        super().__init__(n_var=self.prob.num_dim,
                         n_obj=self.prob.num_obj,
                         xl=np.array([coord_bounds[0] for coord_bounds in self.prob.bounds]),
                         xu=np.array([coord_bounds[1] for coord_bounds in self.prob.bounds]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = self.prob.objective[0](x)
        f2 = self.prob.objective[1](x)

        out["F"] = [f1, f2]

def run_benchmark_pymoo(bproblem):
    pymoo_problem=PymooProblem(bproblem)
    algorithm = NSGA2(
        pop_size=1000,
        n_offsprings=250,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 1000)
    res = minimize(pymoo_problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)

    X = res.X
    F = res.F
    return (X,F)

def run_benchmark_pymoo_with_budget(bproblem,D,Y):
    num_dim=bproblem.num_dim
    n_init=init_factor*num_dim+1
    n_i=min_factor*num_dim+1
    budget=budget_factor*num_dim-n_init-2*n_i
    pymoo_problem=PymooProblem(bproblem)

    X = np.array([x.tolist() for x in D])
    pop = Population.new("X", X)
    Evaluator().eval(pymoo_problem, pop)

    all_evaluated_X = [np.array([ind.X for ind in pop])]
    all_evaluated_F = [np.array([ind.F for ind in pop])]


    algorithm = NSGA2(
        pop_size=budget_factor,
        n_offsprings=1,
        sampling=pop,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )


    termination = get_termination("n_gen", budget)
    res = minimize(pymoo_problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    
    for algorithm_state in res.history:
        offspring = algorithm_state.off
        if offspring is not None and len(offspring) > 0:
            all_evaluated_X.append(offspring.get("X"))
            all_evaluated_F.append(offspring.get("F"))
    
    historical_X = np.vstack(all_evaluated_X)
    historical_F = np.vstack(all_evaluated_F)

    return (historical_X, historical_F)

data=[]

for bproblem in L[:a**2]:
    state,pareto_points_naive,D_naive,Y_naive,X,F,X2,F2=run_all_benchmarks(bproblem)
    data.append((bproblem.name,state,X,F,pareto_points_naive,D_naive,Y_naive,X2,F2))
    data_to_save = {
        "name":bproblem.name,
        "state_biego":state,
        "X":X,
        "F":F,
        "pareto_naive":pareto_points_naive,
        "D_naive":D_naive,
        "Y_naive":Y_naive,
        "X2":X2,
        "F2":F2,
        "parameters":parameters,
    }

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"benchmark_run_{current_time}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(data_to_save, file)

    print(f"Results successfully saved to {filename}")

print(f"Test {test_number}: n_accuracy = {n_accuracy}, seed = {seed}, budget = {budget_factor} * dim, n_init = {init_factor} * dim + 1, n_min = {min_factor} * dim + 1, n_single_objective_max = {max_so_iter_factor} * dim + 1, n_multistart = {multi_start_factor} * dim, single-objective formulation = {soformulation_composite} for composite, {soformulation_naive} for naive")

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

print(L)

for i in range(a):
    for j in range(a):
        index = i*3+j
        num_dim=L[index].num_dim
        ax=axs[i][j]

        D,Y=get_DoE(data[index][1])
        D_naive=data[index][5]
        Y_naive=data[index][6]
        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]
        X2,F2 = data[index][7],data[index][8]

        T=[i for i in range(init_factor*num_dim+1+2*(min_factor*num_dim+1),len(D))]
        IGD_ac=[]
        IGD_naive=[]
        IGD_pymoo_budget=[]
        for t in T:
            pareto_points = [(D[i],Y[i]) for i in ParetoFront(D[:t+1],Y[:t+1])]
            pareto_points_pymoo_budget = [(X2[i],F2[i]) for i in ParetoFront(X2[:t+1],F2[:t+1])]
            pareto_points_naive = [(D_naive[i],Y_naive[i]) for i in ParetoFront(D_naive[:t+1],Y_naive[:t+1])]
            IGD_ac.append(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points])))
            IGD_pymoo_budget.append(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo_budget])))
            IGD_naive.append(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])))

        ax.semilogy(T,IGD_ac,label="composite biEGO")
        ax.semilogy(T,IGD_naive,label="naive biEGO")
        ax.semilogy(T,IGD_pymoo_budget,label="pymoo")
        ax.title.set_text(f"{data[index][0]}")
        ax.legend(loc="best")
        ax.set_xlabel("budget")
        ax.set_ylabel("IGD+")

plt.show()

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

for i in range(a):
    for j in range(a):
        index = i*3+j
        num_dim=L[index].num_dim
        ax=axs[i][j]

        D,Y=get_DoE(data[index][1])
        D_naive=data[index][5]
        Y_naive=data[index][6]
        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]
        X2,F2 = data[index][7],data[index][8]


        T=[i for i in range(init_factor*num_dim+1+2*(min_factor*num_dim+1),len(D))]
        HV_ac=[]
        HV_naive=[]
        HV_pymoo_budget=[]
        ref_point=np.array([1,1])
        for t in T:
            pareto_points = [(D[i],Y[i]) for i in ParetoFront(D[:t+1],Y[:t+1])]
            pareto_points_pymoo_budget = [(X2[i],F2[i]) for i in ParetoFront(X2[:t+1],F2[:t+1])]
            pareto_points_naive = [(D_naive[i],Y_naive[i]) for i in ParetoFront(D_naive[:t+1],Y_naive[:t+1])]
            HV_ac.append(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points])))
            HV_pymoo_budget.append(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo_budget])))
            HV_naive.append(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])))

        ax.plot(T,HV_ac,label="composite biEGO")
        ax.plot(T,HV_naive,label="naive biEGO")
        ax.plot(T,HV_pymoo_budget,label="pymoo")
        ax.title.set_text(f"{data[index][0]}")
        ax.legend(loc="best")
        ax.set_xlabel("budget")
        ax.set_ylabel("Hypervolume")

plt.show()

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

for i in range(a):
    for j in range(a):
        index = i*3+j
        ax=axs[i][j]
        ref_point=np.array([1,1])

        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]
        ax.scatter([p[1][0] for p in pareto_points_pymoo],[p[1][1] for p in pareto_points_pymoo],marker=".",s=5,color="red",label="Optimal Pareto front")

        D,Y=get_DoE(data[index][1])
        pareto_points = [(D[i],Y[i]) for i in ParetoFront(D,Y)]
        ax.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points],color="blue",label="Composite acquisition function")

        pareto_points_naive=data[index][4]
        ax.scatter([p[1][0] for p in pareto_points_naive],[p[1][1] for p in pareto_points_naive],color="green",label="Naive biEGO",marker="+",s=60)

        X2,F2=data[index][7],data[index][8]
        pareto_points_pymoo2 = [(X[i],F[i]) for i in ParetoFront(X2,F2)]
        ax.scatter([p[1][0] for p in pareto_points_pymoo2],[p[1][1] for p in pareto_points_pymoo2],marker="x",color="purple",label="Pymoo with same budget")

        ax.title.set_text(f"""{data[index][0]} IGD+: {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points])),2)} (biEGO), {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])),2)} (naive), {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo2])),2)} (pymoo)\nHypervolume: {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points])),2)} (biEGO), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])),2)} (naive), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo2])),2)} (pymoo), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])),2)} (true front)""")
        ax.legend(loc="best")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")

plt.show()

