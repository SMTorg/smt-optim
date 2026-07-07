import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import math

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies import MFSEGO
from smt_optim.acquisition_strategies.multiobj import MultiObj
from smt_optim.core import Sample
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.utils.constraints import compute_rscv

from tests.test_bisego.custom_driver import CustomStopDriver


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

def Norm(p):
    #Returns the 2-norm of a point
    return np.sqrt(sum(x**2 for x in p))

def Dist(p,q):
    #Returns the distance between two points
    return Norm([p[i]-q[i] for i in range(len(p))])

def DistToNeighbors(p1,p2,p3,w):
    #Returns the sum of squared distances from p2 to its neighbors p1 and p3 on the Pareto front, coefficiented by the Weight.
    return (Dist(p1,p2)**2+Dist(p2,p3)**2)/(w+1)

def PositivePart(x):
    return max(x,0)

def SingleObjectiveProduct(y,r):
    #Returns a single objective product formulation of the problem
    return -math.prod(PositivePart(r[i]-y[i])**2 for i in range(len(y)))

def SingleObjectiveProductModified(y,r):
    #Returns a single objective product formulation of the problem, that is equal to 0 except in the bottom left corner
    if all(y[i]>r[i] for i in range(len(y))):
        return 0
    return -math.prod((r[i]-y[i])*(abs(r[i]-y[i])) for i in range(len(y))) #TODO: Find a suitable formulation

def SingleObjectiveNormalized(y,r,s=None):
    if s==None:
        s=[1]*len(y)
    return max((y[i]-r[i])/s[i] for i in range(len(y)))

def InjectData(state, xt, yt, ct=None):
    """
    Injects matrix-form data (X, Y, and optionally Constraints) into the State.
    """
    # 1. Ensure inputs are 2D arrays
    xt = np.atleast_2d(xt)
    yt = np.atleast_2d(yt)
    
    # Handle constraints
    num_samples = xt.shape[0]
    if ct is None:
        ct = np.empty((num_samples, 0))
    else:
        ct = np.atleast_2d(ct)

    # 2. Add to dataset
    for i in range(num_samples):
        # We compute RSCV because export_as_dict will crash without it
        rscv_val = compute_rscv(ct[i:i+1, :], state.problem.cstr_configs).item()
        
        sample = Sample(
            x=xt[i],
            fidelity=0,
            obj=yt[i],
            cstr=ct[i] if ct.size > 0 else np.array([]),
            eval_time=np.zeros(yt.shape[1] + ct.shape[1]),
            metadata={"rscv": rscv_val, "iter": 0, "budget": 0.0, "fidelity": 0}
        )
        state.dataset.add(sample)

    # Update state metadata if necessary
    state.iter = 0

def InjectBiObjectiveData(state, xt, yt, fidelity_level=0):
    """
    Injects unconstrained bi-objective data into the driver state.
    Expects yt as a list of tuples: [(f1(x1), f2(x1)), (f1(x2), f2(x2)), ...]
    """
    # 1. Convert inputs to clean, predictable NumPy formats
    xt = np.atleast_2d(xt)
    yt = np.array([[y[0][0],y[1][0]] for y in yt])  # Converts list of tuples to a neat (N, 2) array
    
    num_samples = xt.shape[0]

    # 2. Directly loop and inject the samples
    for i in range(num_samples):
        sample = Sample(
            x=xt[i],
            fidelity=fidelity_level,
            obj=yt[i],                      # A 1D array of length 2: [f1, f2]
            cstr=np.array([]),               # No constraints
            eval_time=np.zeros(2),           # Exactly 2 slots for your 2 objectives
            metadata={
                "iter": 0, 
                "budget": 0.0, 
                "fidelity": fidelity_level,
                "rscv":0.0
            }
        )
        state.dataset.add(sample)

    state.iter = 0

def ScaleArray(data):
    factor = np.std(data)
    step = np.mean(data)
    return (data-step)/factor

def SimpleEGO(f,bounds,D,Y,max_iter,strat=MFSEGO,surrogate=SmtAutoModel,strat_kwargs=None,stop_conditions=None):
    #Runs a default implementation of EGO on a single-objective problem
    obj_config = ObjectiveConfig(
        [f],
        type="minimize",
        surrogate=surrogate,
    )

    prob_definition = Problem(
        obj_configs=[obj_config],
        design_space=bounds,            # problem bounds
    )

    xt_init = np.vstack([np.atleast_1d(x) for x in D])
    yt_init = np.vstack([np.atleast_1d(y) for y in Y])

    opt_config = DriverConfig(
        max_iter = max_iter,
        nt_init = 0,
        verbose = True,
        scaling = True,
        seed=42,
    )

    driver = CustomStopDriver(prob_definition, opt_config, strategy=strat,strategy_kwargs=strat_kwargs,stop_conditions=stop_conditions)
    
    InjectData(driver.state, xt_init, yt_init)

    state = driver.optimize()
    return state

def DoubleEGO(f1,f2,bounds,D,Y,max_iter,strat=MultiObj,surrogate=SmtAutoModel,strat_kwargs=None,stop_conditions=None):
    #Runs a default implementation of EGO on a bi-objective problem
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

    xt_init = np.vstack([np.atleast_1d(x) for x in D])
    yt_init = np.array([[y,z] for y,z in Y])

    opt_config = DriverConfig(
        max_iter = max_iter,
        nt_init = 0,
        verbose = True,
        scaling = True,
        seed=42,
    )

    driver = CustomStopDriver(prob_definition, opt_config, strategy=strat,strategy_kwargs=strat_kwargs,stop_conditions=stop_conditions)
    InjectBiObjectiveData(driver.state, xt_init, yt_init)

    state = driver.optimize()
    return state

def StopConditionChangeFront(state,config,D,Y,X):
    X2=ParetoFront([a for a in D],[b for b in Y])
    if X2!=X:
        return False
    return True

def StopConditionBigFront(state,config,D,Y,n):
    X=ParetoFront(D,Y)
    if len(X)>=n:
        return False
    return True

def NaiveBiEGO(F,D,Y,Ng,Ni,Ni0,bounds,n_multistart=5,soformulation="Normalized",show1D=False):
    """
        Find the Pareto front of F:x->(f1(x),f2(x)), with the DoE D=[x1,...,xt] and Y=[F(x1),...,F(xt)],
        using at most Ng evaluations of F. Ni is the maximum number of calls used for any single EGO resolution
    """
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
    f1=lambda x:F_eval(x)[0]
    f2=lambda x:F_eval(x)[1]

    #Run EGO on the problems min(f1(x)) and min(f2(x))
    print("Min(f1) phase")
    state=SimpleEGO(f1,bounds,D,[y[0] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart})
    print("Min(f2) phase")
    state=SimpleEGO(f2,bounds,D,[y[1] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart})
    
    X=ParetoFront(D,Y)

    while n_eval<Ng:
        skip=False
        #1-Determine the reference point
        J=len(X)
        print("The Pareto front is of length",J)
        if J>2:
            #Select a point of the Pareto front relatively far from its neighbors
            j=max((DistToNeighbors(Y[X[k-1]],Y[X[k]],Y[X[k+1]],W[X[k]]),k) for k in range(1,J-1))[1]
            r=(Y[X[j+1]][0],Y[X[j-1]][1])
        elif J==2:
            j=1
            r=(Y[X[1]][0],Y[X[0]][1])
        elif J==1:
            #Run EGO on the problems min(f1(x)) and min(f2(x)) until there are at least two and 3 points in the Pareto front.
            print("Min(f1) phase")
            state=SimpleEGO(f1,bounds,D,[y[0] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart},stop_conditions=[(StopConditionBigFront,{"D":D,"Y":Y,"n":2})])
            print("Min(f2) phase")
            SimpleEGO(f2,bounds,D,[y[1] for y in Y],Ni0,MFSEGO,strat_kwargs={"n_start":n_multistart},stop_conditions=[(StopConditionBigFront,{"D":D,"Y":Y,"n":3})])
            skip=True
        else:
            raise ValueError("The Pareto Front is empty")

        if not skip:
            print("Bi-objective phase with r =",r)
            #2-Run EGO on the single-objective sub-problem
            if soformulation=="Normalized":
                phi = lambda y: SingleObjectiveNormalized(y,r)
            elif soformulation=="Product":
                phi = lambda y: SingleObjectiveProductModified(y,r)
            else:
                raise ValueError("Unknown single-objective formulation")

            def SubProblem(x):
                y=F_eval(x)
                return phi(y)

            if show1D:
                _,ax=plt.subplots(1,1)
                x = np.linspace(bounds[0][0], bounds[0][1], 500)
                y1=F(x)[0]
                y2=F(x)[1]
                ax.plot(y1,y2,color='gray', alpha=0.3, linestyle='--', label='Feasible objective range')
                ax.scatter([r[0]],[r[1]])
                ax.scatter([y[0] for y in Y],[y[1] for y in Y])
                ax.scatter([Y[i][0] for i in X],[Y[i][1] for i in X])
                ax.scatter([Y[-1][0]],[Y[-1][1]],color="red")
                plt.show()

            state=SimpleEGO(SubProblem,bounds,D,[phi(y) for y in Y],Ni,MFSEGO,strat_kwargs={"n_start":n_multistart},surrogate=SmtAutoModel,stop_conditions=[(StopConditionChangeFront,{"D":D,"Y":Y,"X":X})])

            #3-Update Weights
            W[X[j]]+=1
        
        #Update Pareto front
        X=ParetoFront(D,Y)

    X=ParetoFront(D,Y)
    return ([(D[i],Y[i]) for i in X],D,Y)

def AcBiEGO(F,D,Y,Ng,Ni,Ni0,bounds,soformulation="Normalized",show1D=False):
    """
        Find the Pareto front of F:x->(f1(x),f2(x)), with the DoE D=[x1,...,xt] and Y=[F(x1),...,F(xt)],
        using at most Ng evaluations of F. Ni is the maximum number of calls used for any single EGO resolution
        This function uses an adapted composite acquisition function
    """
    #Initialization
    print("biEGO Step 1")
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
    f1=lambda x:F_eval(x)[0]
    f2=lambda x:F(x)[1] # /!\ TEMPORARY FIX We suppose that f1 is always called when f2 is called, but the current implementation doubles the number of calls to F = (f1,f2), to solve that we need to either separate f1 and f2 or modify the driver evaluation for multi-objective functions
    f2_eval=lambda x:F_eval(x)[1]

    #Run EGO on the problems min(f1(x)) and min(f2(x))
    print("Running MFSEGO on min(f1)")
    state=SimpleEGO(f1,bounds,D,[y[0] for y in Y],Ni0)
    print("Running MFSEGO on min(f2)")
    state=SimpleEGO(f2_eval,bounds,D,[y[1] for y in Y],Ni0)
    
    X=ParetoFront(D,Y)

    while n_eval<Ng:
        print("biEGO Step 2")
        skip=False
        #1-Determine the reference point
        J=len(X)
        print("The Pareto front is of length",J)
        if J>2:
            #Select a point of the Pareto front relatively far from its neighbors
            j=max((DistToNeighbors(Y[X[k-1]],Y[X[k]],Y[X[k+1]],W[X[k]]),k) for k in range(1,J-1))[1]
            r=(Y[X[j+1]][0],Y[X[j-1]][1])
        elif J==2:
            j=1
            r=(Y[X[1]][0],Y[X[0]][1])
        elif J==1:
            #Run EGO on the problems min(f1(x)) and min(f2(x)) until there are at least two and 3 points in the Pareto front.
            print("Running MFSEGO on min(f1)")
            state=SimpleEGO(f1,bounds,D,[y[0] for y in Y],Ni0,stop_conditions=[(StopConditionBigFront,{"D":D,"Y":Y,"n":2})])
            print("Running MFSEGO on min(f2)")
            state=SimpleEGO(f2_eval,bounds,D,[y[1] for y in Y],Ni0,stop_conditions=[(StopConditionBigFront,{"D":D,"Y":Y,"n":3})])
            skip=True
        else:
            raise ValueError("The Pareto Front is empty")

        if not skip:
            print("biEGO step 3")
            #2-Run EGO on the single-objective sub-problem
            if soformulation=="Normalized":
                phi = lambda y: SingleObjectiveNormalized(y,r)
            elif soformulation=="Product":
                phi = lambda y: SingleObjectiveProduct(y,r)
            else:
                raise ValueError("Unknown single-objective formulation")

            def build_composite_expected_improvement(state,phi=phi):

                def composite_expected_improvement(mu: float, s2: float, f_min: float, n_expectancy=1000) -> float:
                    """
                    Expected Improvement composite acquisition function.

                    Parameters
                    ----------
                    mu: np.array
                        Mean prediction.
                    s2: np.array
                        Variance prediction.
                    f_min: float
                        Best minimum objective value in training data.
                    phi: np.array -> float

                    Returns
                    -------
                    float
                        Expected Improvement value.
                    """

                    S=np.atleast_1d(0.0)
                    for i in range(n_expectancy):
                        sampleZ = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,1]]))
                        S+=PositivePart(f_min-phi(mu+s2*sampleZ))
                    ei=S/n_expectancy

                    return ei[0]
                
                models=state.obj_models
                f_min=min([phi(y) for y in state.scaled_dataset.export_data([0,1],0)])

                def cei(x_pred):
                    s = np.array([
                        np.sqrt(models[0].predict_variances(x_pred)).item(),
                        np.sqrt(models[1].predict_variances(x_pred)).item()
                    ])

                    y = np.array([
                        models[0].predict_values(x_pred).item(),
                        models[1].predict_values(x_pred).item(),
                    ])
                    return composite_expected_improvement(y,s,f_min)
                return cei

            print("Running Multi-EGO on the biobjective subproblem with specific acquisition function")
            print("r",r)
            state=DoubleEGO(f1,f2,bounds,D,Y,Ni,MultiObj,strat_kwargs={"acq_func":build_composite_expected_improvement},stop_conditions=[(StopConditionChangeFront,{"D":D,"Y":Y,"X":X})])

            if show1D:
                _,axs=plt.subplots(1,3,figsize=(15, 5))
                ax1,ax2,ax3=axs[0],axs[1],axs[2]
                x = np.linspace(bounds[0][0], bounds[0][1], 500)
                y1=F(x)[0]
                y2=F(x)[1]
                ax1.plot(y1,y2,color='gray', alpha=0.3, linestyle='--', label='Feasible objective range')
                ax1.scatter([r[0]],[r[1]])
                ax1.scatter([y[0] for y in Y],[y[1] for y in Y])
                ax1.scatter([Y[i][0] for i in X],[Y[i][1] for i in X])
                ax1.scatter([Y[-1][0]],[Y[-1][1]],color="red")

                x2=np.linspace(0,1,500)
                model1=state.obj_models[0].predict_values(x2)
                model2=state.obj_models[1].predict_values(x2)

                y1_exp0=Y[0][0]
                y2_exp0=Y[0][1]
                y1_exp1=Y[Ni][0]
                y2_exp1=Y[Ni][1]

                a=bounds[0][0]
                b=bounds[0][1]
                def affine_scaling(x):
                    return (x-a)/(b-a)

                y1_model0=state.obj_models[0].predict_values(affine_scaling(D[0]))
                y2_model0=state.obj_models[1].predict_values(affine_scaling(D[0]))
                y1_model1=state.obj_models[0].predict_values(affine_scaling(D[Ni]))
                y2_model1=state.obj_models[1].predict_values(affine_scaling(D[Ni]))


                alpha1=(y1_exp1-y1_exp0)/(y1_model1-y1_model0)
                beta1=y1_exp0-alpha1*y1_model0

                model1=model1*alpha1+beta1

                alpha2=(y2_exp1-y2_exp0)/(y2_model1-y2_model0)
                beta2=y2_exp0-alpha2*y2_model0

                model2=model2*alpha2+beta2

                ax2.plot(x,y1,color='gray', alpha=0.3, label='Actual function')
                ax2.plot(x,model1,color='green', alpha=0.3, label='Acquisition function')

                
                ax3.plot(x,y2,color='gray', alpha=0.3, label='Actual function')
                ax3.plot(x,model2,color='green', alpha=0.3, label='Acquisition function')

                cei=build_composite_expected_improvement(state)
                colorvalue=[cei(np.atleast_1d(x)) for x in x2]


                m1_flat = np.asarray(model1).ravel()
                m2_flat = np.asarray(model2).ravel()
                points = np.vstack((m1_flat, m2_flat)).T.reshape(-1, 1, 2)
                segments = np.hstack((points[:-1], points[1:]))
                lc = LineCollection(segments, array=colorvalue, cmap="jet", lw=3)
                line = ax1.add_collection(lc)
                ax1.autoscale()
                plt.show()

            #3-Update Weights
            print("biEGO step 4")
            W[X[j]]+=1
        
        #Update Pareto front
        print("Updating Pareto front")
        X=ParetoFront(D,Y)
        print("Pareto front:",X)

    X=ParetoFront(D,Y)
    return ([(D[i],Y[i]) for i in X],state)


def test_poly(x):
    return -np.atleast_1d(3*x**5-10*x**3+2*x**2-7*x+1)

def gaussian(x,mu=np.array([1.1]),v=np.array([2])):
    return -np.exp(-(x-mu)**2/(v**2))

def sasena_2002(x: np.ndarray):
    return -np.sin(x) - np.exp(x / 100)

def sasena_bis(x):
    return -np.cos(x) + np.exp(x/100)

def quad(x: np.ndarray):
    return np.atleast_1d(x**2)

def quad2(x: np.ndarray):
    return np.atleast_1d((x-2)**2)

def weierstrass(x,a=3,b=0.5,Nvar=100):
    we=0
    for n in range(0,Nvar):
        we=we+np.cos(a**n*np.pi*x)*b**n
    return -we

def absxsinsurx(x,c=1.3):
    if x[0]==c:
        return np.atleast_1d(0)
    return (-2*abs(x-c)+(x-c)*np.sin(1/(x-c)))

def Fonseca_Fleming1(x):
    return np.atleast_1d(1-np.exp(-np.sum((x-1/(np.sqrt(len(x))))**2)))

def Fonseca_Fleming2(x):
    return np.atleast_1d(1-np.exp(-np.sum((x+1/(np.sqrt(len(x))))**2)))

def easy(x):
    return np.atleast_1d(x[0]+x[1])

def easy2(x):
    return np.atleast_1d((x[0]+x[1])**2)

def main():
    print("--- Starting Bi-Objective EGO Optimization Test ---")

    f1=Fonseca_Fleming1
    f2=Fonseca_Fleming2

    def F_target(val):
        return (f1(val), f2(val))

    bounds = np.array([[-4, 4],[-4,4]])

    D = [np.array([-1.0,0.0]),np.array([1.0,1.3]), np.array([0.1,-3.0]),np.array([-2.0,-0.5]),np.array([-3.0,1.0])]
    Y = [F_target(x) for x in D]

    Ng = 35  # Total budget of extra evaluations
    Ni = 5   # Max iterations per single-objective EGO sub-call
    Ni0 = 5 # Max iterations per call to min(f1) or min(f2)

    pareto_points,state = AcBiEGO(
        F=F_target,
        D=D,
        Y=Y,
        Ng=Ng,
        Ni=Ni,
        Ni0=Ni0,
        bounds=bounds,
        soformulation="Product",
        show1D=False
    )

    print("\n--- Optimization Complete ---")
    print(f"Number of points on the Pareto Front: {len(pareto_points)}")
    print("Pareto Optimal X values:")
    for pt in [p[0] for p in pareto_points]:
        print(f"  x: {pt}, F(x): {F_target(pt)}")
    
    _,ax1=plt.subplots(1,1,figsize=(15, 5))
    ax1.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points])

    D = [np.array([-1.0,0.0]),np.array([1.0,1.3]), np.array([0.1,-3.0])]
    Y = [F_target(x) for x in D]
    
    pareto_points,state = NaiveBiEGO(
        F=F_target,
        D=D,
        Y=Y,
        Ng=Ng,
        Ni=Ni,
        Ni0=Ni0,
        bounds=bounds,
        soformulation="Product",
        show1D=False
    )

    print("\n--- Optimization Complete ---")
    print(f"Number of points on the Pareto Front: {len(pareto_points)}")
    print("Pareto Optimal X values:")
    for pt in [p[0] for p in pareto_points]:
        print(f"  x: {pt}, F(x): {F_target(pt)}")
            

    ax1.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points])

    plt.show()


if __name__ == "__main__":
    main()


