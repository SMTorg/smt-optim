import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as so, stats as stats

from smt_optim.acquisition_functions import log_ei
from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core.state import State

from smt_optim.utils.get_fmin import get_fmin

from smt_optim.subsolvers import multistart_minimize

from smt_optim.acquisition_functions.multi_obj import init_bi_obj_composite_ei

from scipy.stats import norm
from smt.surrogate_models import KRG

def Dominates(p,q):
    #Returns True if point p strictly dominates point q, else returns False
    return (p[0]<=q[0] and p[1]<q[1]) or (p[0]<q[0] and p[1]<=q[1])

def ParetoFront(D,Y):
    #Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    t=len(D)
    front=[]
    for i in range(t):
        if all([not Dominates(q,Y[i]) for q in Y]):
            front.append((Y[i][0],i))
    front.sort()
    return [p[1] for p in front]

def PositivePart(x):
    return max(x,0)

def SingleObjectiveNormalized(y,r,s=None):
    if s==None:
        s=[1]*len(y)
    return max((y[i]-r[i])/s[i] for i in range(len(y)))

def SingleObjectiveProduct(y,r):
    #Returns a single objective product formulation of the problem
    return -np.prod([PositivePart(r[i]-y[i])**2 for i in range(len(y))])

def Norm(p):
    #Returns the 2-norm of a point
    return np.sqrt(sum(x**2 for x in p))

def Dist(p,q):
    #Returns the distance between two points
    return Norm([p[i]-q[i] for i in range(len(p))])

def DistToNeighbors(p1,p2,p3,w):
    #Returns the sum of squared distances from p2 to its neighbors p1 and p3 on the Pareto front, coefficiented by the Weight.
    return (Dist(p1,p2)**2+Dist(p2,p3)**2)/(w+1)

class BiSEGO(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()


        self.acq_func1 = kwargs.get("acq_func", log_ei) #Acquisition function for min(f1) (to be modified to take only f1 as a parameter)
        self.acq_func2 = kwargs.get("acq_func", log_ei) #Acquisition function for min(f2) (same for f2)
        self.acq_func_gen3 = kwargs.get("acq_func_bi", init_bi_obj_composite_ei) #Composite acquisition function for min(f1,f2)
        self.acq_func_gen_naive = kwargs.get("acq_func_naive", log_ei) #Naive acquisition function for min(phi(f1,f2))
        self.n_multi_start = kwargs.pop("n_multi_start", 5)
        self.n_accuracy = kwargs.pop("n_accuracy",1000)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))
        self.soformulation=kwargs.pop("so_formulation","Product")
        self.current_calls = 0
        self.current_subcalls = 0
        self.n_init = kwargs.pop("n_init",self.n_multi_start)
        self.single_obj_max_calls = kwargs.pop("single_obj_max_calls",self.n_init)
        self.min_max_calls = kwargs.pop("min_max_calls",self.n_init)
        self.acq_func_gen1 = lambda state,kwargs : lambda x : self.acq_func1(state.obj_models[0].predict_values(x),state.obj_models[0].predict_variances(x),min(state.scaled_dataset.export_data([0],0)))[0][0]
        self.acq_func_gen2 = lambda state,kwargs : lambda x : self.acq_func2(state.obj_models[1].predict_values(x),state.obj_models[1].predict_variances(x),min(state.scaled_dataset.export_data([1],0)))[0][0]
        self.naive = kwargs.pop("naive",False)

        self.r = None
        self.state = state
        self.X = None
        self.W = None


    def validate_config(self, state):
        pass

    def get_scaled_DoE(self):
        Y=self.state.scaled_dataset.export_as_dict()["obj"]
        D=self.state.scaled_dataset.export_as_dict()["x"]
        return (D,Y)

    def get_DoE(self):
        Y=self.state.dataset.export_as_dict()["obj"]
        D=self.state.dataset.export_as_dict()["x"]
        return (D,Y)
    
    def get_pareto_front(self):
        D,Y=self.get_scaled_DoE()
        self.X = ParetoFront(D,Y)

    def select_reference_point(self):
        self.get_pareto_front()
        J=len(self.X)
        D,Y=self.get_scaled_DoE()
        X=self.X
        W=self.W
        if J>2:
            #Select a point of the Pareto front relatively far from its neighbors
            j=max((DistToNeighbors(Y[X[k-1]],Y[X[k]],Y[X[k+1]],W[X[k]]),k) for k in range(1,J-1))[1]
            r=(Y[X[j+1]][0],Y[X[j-1]][1])
        elif J==2:
            j=1
            r=(Y[X[1]][0],Y[X[0]][1])
        elif J==1:
            return None
        else:
            raise ValueError("The Pareto Front is empty")
        self.W[X[j]]+=1
        return r

    def get_infill(self, state):
        old_pareto_front=self.X
        self.get_pareto_front()

        if self.current_calls == 0:
            if self.current_calls < self.min_max_calls:
                print("Min(f1) phase")
            self.W = [0 for x in range(len(self.state.dataset.export_as_dict()["x"]))]

        # Pre-treatment phase : find min(f1) and min(f2)
        if self.current_calls < self.min_max_calls:
            self.current_calls+=1
            self.W.append(0)
            return self.get_infill_custom(state,self.acq_func_gen1)
        elif self.current_calls < 2*self.min_max_calls:
            if self.current_calls==self.min_max_calls:
                print("Min(f2) phase")
            self.current_calls+=1
            self.W.append(0)
            return self.get_infill_custom(state,self.acq_func_gen2)

        # Main loop
        else:
            if self.current_calls == 2*self.min_max_calls or self.current_subcalls == 0 or self.current_subcalls == self.single_obj_max_calls or old_pareto_front!=self.X :
                print("The Pareto front is of length", len(self.X))
                self.current_subcalls = 0
                r=self.select_reference_point()
                if r==None: # edge case where there is only one point in the Pareto Front
                    self.current_subcalls+=1
                    self.current_calls+=1
                    self.W.append(0)
                    if self.current_calls%2:
                        return self.get_infill_custom(state,self.acq_func_gen1)
                    else:
                        return self.get_infill_custom(state,self.acq_func_gen2)
                print("Bi-objective phase with r =",r)
                if self.soformulation=="Normalized":
                    self.phi = lambda y: SingleObjectiveNormalized(y,r)
                elif self.soformulation=="Product":
                    self.phi = lambda y: SingleObjectiveProduct(y,r)
                else:
                    raise ValueError("Unknown single-objective formulation")
            self.current_subcalls+=1
            self.current_calls+=1
            self.W.append(0)

            if not self.naive:
                return self.get_infill_custom(state,self.acq_func_gen3,phi=self.phi,n_accuracy=self.n_accuracy)
            else:
                return self.get_infill_naive(state,self.phi)

    def get_infill_custom(self,state,acq_func_gen,**kwargs):
        # Gets the infill point using the custom acquisition function provided
        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_multi_start)

        ac_func = acq_func_gen(state,kwargs)

        def sp_wrapper(x):
            x = x.reshape(1, -1)
            return -ac_func(x)

        res = multistart_minimize(sp_wrapper,
                                    bounds=np.array([[0, 1]] * state.problem.num_dim),
                                    constraints=[],
                                    n_start=self.n_multi_start,
                                    multi_x0=multi_x0,
                                    seed=self.seed,
                                    tol=self.sp_tol,
                                    method=self.sp_method, )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill

    def get_infill_naive(self,state,phi):

        D,Y=self.get_scaled_DoE()
        Phi_values=[phi(y) for y in Y]

        # 1. Train model on DoE
        # Initialize the model
        sm = KRG(print_global=False)
        sm.set_training_values(np.atleast_1d(D), np.atleast_1d(Phi_values))
        sm.train()

        # Find the current minimum value in the observation history
        current_best = np.min(Y)

        # Define the negative log Expected Improvement to minimize
        def log_ei(x):
            x_reshaped = np.atleast_2d(x)
            mu = sm.predict_values(x_reshaped)[0, 0]
            sigma2 = sm.predict_variances(x_reshaped)[0, 0]
            sigma = np.sqrt(max(sigma2, 1e-12))
            improvement = current_best - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.log(max(ei, 1e-12))

        # 2. Find maximum of log_EI acquisition function
        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_multi_start)

        def sp_wrapper(x):
            x = x.reshape(1, -1)
            return -log_ei(x)

        res = multistart_minimize(sp_wrapper,
                                    bounds=np.array([[0, 1]] * state.problem.num_dim),
                                    constraints=[],
                                    n_start=self.n_multi_start,
                                    multi_x0=multi_x0,
                                    seed=self.seed,
                                    tol=self.sp_tol,
                                    method=self.sp_method, )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill
    
