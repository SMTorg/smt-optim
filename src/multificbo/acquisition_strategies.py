import numpy as np
import scipy.stats as stats
import scipy.optimize as so
from warnings import warn

from deap import base, creator, tools, algorithms
import random

from multificbo.surrogate_models import Surrogate
from multificbo.acquisition_functions import expected_improvement, log_ei, probability_of_improvement, fidelity_correlation



# TODO: acquisition strategy class template
class AcquisitionStrategy:
    def __init__(self):
        pass

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    def execute_infill_strategy(self, optimizer):
        raise Exception("Acquisition Strategy not implemented.")


class MonoFiAcqStrat(AcquisitionStrategy):
    def __init__(self, acq_func=log_ei, optimizer=None):
        super().__init__()

        self.acq_func = acq_func
        self.n_start = 10

        if optimizer is not None:
            self.n_start *= optimizer.num_dim

    def execute_infill_strategy(self, optimizer) -> np.ndarray:
        return self.acq_strategy(optimizer)


    def acq_strategy(self, optimizer) -> np.ndarray:
        """
        Maximize the expected improvement acquisition function to find the next infill location. Handle constraints if
        applicable. The optimization is done with scipy's COBYLA implementation.

        Args:
            f_min (float):  best feasible objective value sampled so far
            bounds (np.ndarray): variable bounds
            obj_surrogate (Surrogate):  objective function surrogate model
            cstr_surrogates (list[Surrogate]):  list of constraint function surrogate models

        Returns:
            next_x (np.ndarray): coordinates of the next infill location
        """

        dim = optimizer.num_dim
        acq_multistart = self.n_start

        f_min = optimizer.f_min
        bounds = optimizer.domain

        num_cstr = optimizer.num_cstr

        obj_surrogate = optimizer.obj_surrogate
        cstr_surrogates = optimizer.cstr_surrogates


        ct_norm = np.linalg.norm(optimizer.ct[-1], axis=1)

        # if there is no feasible points in the training data, f_min is set to objective value corresponding to the
        # minimum constraint violation
        if f_min == np.inf:
            argmin_ct_norm = np.argmin(ct_norm)
            f_min = optimizer.yt[-1][argmin_ct_norm]

        def scipy_ei_wrapper(x):

            x = x.reshape(1, -1)

            mu = obj_surrogate.predict_values(x)
            s2 = obj_surrogate.predict_variances(x)

            return -self.acq_func(mu, s2, f_min)

        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        acq_res_x = np.empty_like(acq_x0)
        acq_res_f = np.empty(acq_multistart)
        acq_res_c = np.empty((acq_multistart, num_cstr))

        scipy_cstr = []
        if num_cstr > 0:
            scipy_cstr = [{"type": "ineq", "fun": lambda x, c_gp=c_gp: -c_gp.predict_values(x.reshape(1, -1)).ravel()} for c_gp in
                          cstr_surrogates]

        for i in range(acq_x0.shape[0]):
            so_x0 = acq_x0[i, :]
            acq_res = so.minimize(scipy_ei_wrapper, so_x0, bounds=bounds, method="COBYLA", constraints=scipy_cstr, tol=1e-4)
            acq_res_x[i, :] = acq_res.x
            acq_res_f[i] = acq_res.fun

            for c_id in range(num_cstr):
                acq_res_c[i, c_id] = cstr_surrogates[c_id].predict_values(acq_res_x[i, :].reshape(1, -1))

        # check if solution respect the constraints
        feas_mask = np.all(acq_res_c <= 1e-4, axis=1)   # TODO: add user parameter to modify the tolerance
        if np.any(feas_mask):
            next_index = np.argmin(
                np.where(feas_mask == True, acq_res_f, np.inf)
            )
        else:
            # if none of the solutions are feasible, the solution with the lowest constraint violation is selected
            acq_res_c_norm = np.linalg.norm(acq_res_c, axis=1)
            next_index = np.argmin(acq_res_c_norm)
            warn("No feasible point found through the acquisition function.")

        next_x = acq_res_x[next_index, :]

        return next_x


class MultiFiAcqStrat(AcquisitionStrategy):
    def __init__(self, acq_func=log_ei, optimizer=None):
        super().__init__()

        self.acq_func = acq_func
        self.n_start = 10

        self.mono_fi_strat = MonoFiAcqStrat(acq_func=acq_func, optimizer=optimizer)

        if optimizer is not None:
            self.n_start *= optimizer.num_dim


    def execute_infill_strategy(self, optimizer) -> list:
        if optimizer.num_levels == 1:
            raise Exception("Incorrect acquisition strategy. Problem is not multi-fidelity.")

        return self.mf_acq_strategy(optimizer)

    def mf_acq_strategy(self, optimizer) -> list:

        # get next sampling point
        next_x = self.mono_fi_strat.execute_infill_strategy(optimizer)

        n_level = optimizer.num_levels

        # TODO: implement the different multi-fidelity strategies
        obj_surrogate = optimizer.obj_surrogate
        obj_cost = obj_surrogate.costs

        # Get the variance reduction and the associated scaling factor of each level
        # TODO: function should only return the variance -> variance reduction computatio should be done here
        s2_red, rho2 = obj_surrogate.predict_s2_red_rho2(next_x.reshape(1, -1))

        tot_cost = np.zeros(n_level)
        fid_crit = np.zeros(n_level)

        # Compute the highest fidelity level that should be sampled
        for k in range(0, n_level):
            tot_cost[k] = np.sum(obj_cost[0:k+1])**2

        norm_tot_cost = tot_cost / np.max(tot_cost)

        fid_crit = s2_red / norm_tot_cost

        optimizer.iter_data["s2_red"] = s2_red
        optimizer.iter_data["tot_cost2"] = tot_cost
        optimizer.iter_data["norm_tot_cost2"] = norm_tot_cost
        optimizer.iter_data["fid_crit"] = fid_crit

        max_level = np.argmax(fid_crit)

        optimizer.iter_data["infill_x"] = next_x
        optimizer.iter_data["infill_max_level"] = max_level    # redundant

        # for each index    -> None:            do not sample level
        #                   -> np.ndarray():    sample level at said point
        next_x_all_lvl = [None] * n_level

        for k in range(max_level+1):
            if k <= max_level:
                next_x_all_lvl[k] = next_x

        return next_x_all_lvl

class MFEI:
    def __init__(self, acq_func=expected_improvement, optimizer=None):
        super().__init__()

        self.optimizer = optimizer
        self.acq_func = acq_func
        self.n_start = 10

    def alpha1(self, x: np.ndarray, l: int, m:int ):

        # or approximate a pearson correlation factor?

        mfk = self.optimizer.obj_surrogate.mfk

        s2_pred, rho2 = mfk.predict_variances_all_levels(x)

        var_l = s2_pred[:, l]
        var_m = s2_pred[:, m]

        kappa = var_l

        for ll in range(l, m):
            kappa *= np.sqrt(rho2[ll])

        alpha1 = kappa/np.sqrt(var_l * var_m)
        # print(f"alpha1 = {alpha1}")
        #
        # mu_l = mfk._predict_intermediate_values(x, l+1)
        # print(f"mu_l = {mu_l}")
        # mu_mm = mfk._predict_intermediate_values(x, m+1)
        # print(f"mu_mm = {mu_mm}")
        # mu_m = mfk._predict_values(x)
        # print(f"mu_m = {mu_m}")
        #
        # samples_l = np.random.normal(loc=mu_l.squeeze(), scale=np.sqrt(var_l.squeeze()), size=10_000)
        # samples_m = np.random.normal(loc=mu_m.squeeze(), scale=np.sqrt(var_m.squeeze()), size=10_000)
        #
        # print(f"pearson = \n{np.corrcoef(samples_l, samples_m)}")
        #
        #
        # print(f"corr = {np.sqrt(var_l.squeeze())/np.abs(mu_m - mu_l) + np.sqrt(var_m)}")
        # apply correction if correlation coefficient is > 1. Should raise warning?
        alpha1 = np.where(alpha1 > 1, 1, alpha1)
        return alpha1

    def alpha2(self, x: np.ndarray, l: int):
        # assume no noise
        return 1.0

    def alpha3(self, l):

        costs = self.optimizer.obj_surrogate.cost

        tot_cost_m = np.sum(costs)
        tot_cost_l = np.sum(costs[0:l+1])

        return tot_cost_m/tot_cost_l


    def execute_infill_strategy(self, optimizer) -> list:

        obj_surrogate = optimizer.obj_surrogate
        dim = optimizer.num_dim
        n_level = optimizer.num_levels
        f_min = optimizer.f_min
        bounds = optimizer.domain

        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(self.n_start)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        best_x_per_level = []
        best_f_per_level = []

        for lvl in range(n_level):

            alpha3 = self.alpha3(lvl)

            acq_res_x = np.empty_like(acq_x0)
            acq_res_f = np.empty(self.n_start)

            def scipy_ei_wrapper(x):
                x = x.reshape(1, -1)

                mu = obj_surrogate.predict_values(x)
                s2 = obj_surrogate.predict_variances(x)

                return -self.acq_func(mu, s2, f_min) * self.alpha1(x, lvl, n_level - 1) * alpha3

            cstr_surrogates = optimizer.cstr_surrogates
            constrained = optimizer.constrained

            if cstr_surrogates is None or cstr_surrogates == []:
                constrained = True

            scipy_cstr = []
            if constrained:
                scipy_cstr = [{"type": "ineq", "fun": lambda x, c_gp=c_gp: -c_gp.predict_values(x.reshape(1, -1))} for
                              c_gp in
                              cstr_surrogates]

            for i in range(acq_x0.shape[0]):

                so_x0 = acq_x0[i, :]
                acq_res = so.minimize(scipy_ei_wrapper, so_x0,
                                      bounds=bounds,
                                      method="COBYLA",
                                      constraints=scipy_cstr,
                                      tol=1e-4)

                acq_res_x[i, :] = acq_res.x
                acq_res_f[i] = acq_res.fun

            index = np.argmin(acq_res_f)
            best_x_per_level.append( acq_res_x[index, :] )
            best_f_per_level.append( acq_res_f[index] )

            optimizer.iter_data[f"lvl{lvl}_acq_f"] = acq_res_f[index]
            optimizer.iter_data[f"lvl{lvl}_alpha1"] = self.alpha1(np.array([acq_res_x[index, :]]), lvl, n_level-1)
            optimizer.iter_data[f"lvl{lvl}_alpha3"] = alpha3

        next_level = np.argmax(best_f_per_level)
        next_x = best_x_per_level[next_level]

        optimizer.iter_data["infill_x"] = next_x
        optimizer.iter_data["infill_max_level"] = next_level

        next_x_all_lvl = [None] * n_level

        for k in range(next_level+1):
            if k <= next_level:
                next_x_all_lvl[k] = next_x

        return next_x_all_lvl


class VFPI:

    def __init__(self, acq_func=expected_improvement, optimizer=None):
        super().__init__()

        self.optimizer = optimizer
        self.acq_func = acq_func
        self.n_start = 10

        self.f_min = np.inf

        self.sub_optimizer = "COBYLA"

        if self.optimizer is not None:
            self.mfck = self.optimizer.obj_surrogate.mfck
            self.n_start *= self.optimizer.num_dim
        else:
            self.mfck = None

    def predicted_f_min(self, level):

        dim = self.optimizer.num_dim
        bounds = self.optimizer.domain
        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        all_f_min = []
        all_x_min = []

        def f_min_scipy_wrapper(x: np.ndarray):
            x = x.reshape(1, dim)
            means, vars = self.mfck.predict_all_levels(x)
            return means[level].ravel()

        for i in range(acq_x0.shape[0]):

            res = so.minimize(f_min_scipy_wrapper,
                              acq_x0[i, :],
                              method="COBYLA",
                              bounds=bounds)

            all_f_min.append(res.fun)
            all_x_min.append(res.x)

        f_min_index = np.argmin(all_f_min)
        f_min = all_f_min[f_min_index]
        x_min = all_x_min[f_min_index]

        return f_min, x_min


    def fidelity_correlation(self, covariance, li_var, lj_var):
        corr = np.clip(abs(covariance / np.sqrt(li_var * lj_var)), 0, 1)
        return corr

    def sample_density(self, x: np.ndarray, lvl: int, mfck):

        x_scale = mfck.X_scale
        x_offset = mfck.X_offset

        x = (x - x_offset) / x_scale

        xt_lvl = mfck.X[lvl]
        xt_lvl = (xt_lvl - x_offset)/x_scale
        dim = xt_lvl.shape[1]

        optimal_theta = mfck.optimal_theta

        if lvl == 0:
            sigma2 = optimal_theta[0]
            theta = optimal_theta[1:dim+1]
        else:
            start = (dim+1)+(2+dim)*(lvl-1)
            end = (dim+1)+(2+dim)*(lvl)
            sigma2 = optimal_theta[start]
            theta = optimal_theta[start+1:end-1]

        R = 1 - mfck._compute_K(x, xt_lvl, (sigma2, theta)) / sigma2
        penalty = np.prod(R, axis=1).reshape(-1, 1)

        return penalty

    def epi(self, x, level) -> np.ndarray:

        x = x.reshape(1, -1)

        means, vars = self.mfck.predict_all_levels(x)
        cov = self.mfck.predict_level_covariances(x, level)

        # probability of improvement
        pi = probability_of_improvement(means[-1].reshape(-1, 1), vars[-1].reshape(-1, 1), self.f_min)

        # fidelity correlation penalty
        corr = fidelity_correlation(cov, vars[level].reshape(-1, 1), vars[-1].reshape(-1, 1))

        # cost ratio penalty
        cost_ratio = self.costs[-1]/self.costs[level]

        # density penalty
        density = self.sample_density(x, level, self.mfck)

        # constraints penalty
        satisfying = 1.0

        # TODO: should use the different fidelity level predictions
        for c_id in range(self.num_cstr):
            g_pred = self.optimizer.cstr_surrogates[c_id].predict_values(x)
            s2_pred = self.optimizer.cstr_surrogates[c_id].predict_variances(x)

            satisfying *= stats.norm.cdf(g_pred / np.sqrt(s2_pred))

        return pi * corr * cost_ratio * density * satisfying

    def execute_infill_strategy(self, optimizer) -> list:

        self.current_level = 0
        self.f_min, _ = self.predicted_f_min(-1)

        self.num_dim = optimizer.num_dim
        self.bounds = optimizer.domain

        self.num_cstr = optimizer.num_cstr

        self.num_levels = optimizer.num_levels
        self.costs = optimizer.costs

        self.optimizer = optimizer

        if self.sub_optimizer == "COBYLA":
            x_infill, fid_infill = self.minimize_with_scipy()

        elif self.sub_optimizer == "DEAP":
            x_infill, fid_infill = self.minimize_with_deap(self.epi,
                                                           self.bounds,
                                                           self.num_levels)
        else:
            raise Exception(f"Unknown sub optimizer: {self.sub_optimizer}")

        infill = []
        for lvl in range(self.num_levels):
            if lvl == fid_infill:
                infill.append(x_infill)
            else:
                infill.append(None)

        return infill


    def minimize_with_scipy(self) -> tuple:

        self.current_level = 0

        def epi_scipy_wrapper(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            epi = self.epi(x, self.current_level).ravel()
            return -epi

        dim = self.optimizer.num_dim
        bounds = self.optimizer.domain
        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=self.num_dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, self.bounds[:, 0], self.bounds[:, 1])

        all_epi_f = []
        all_epi_x = []
        all_epi_fid = []

        for lvl in range(self.num_levels):
            self.current_level = lvl

            for i in range(acq_x0.shape[0]):

                res = so.minimize(epi_scipy_wrapper,
                                  acq_x0[i, :],
                                  method="COBYLA",
                                  bounds=bounds,
                                  tol=1e-4)

                all_epi_f.append(res.fun)
                all_epi_x.append(res.x)
                all_epi_fid.append(self.current_level)


        epi_max_index = np.argmin(all_epi_f)
        x_infill = all_epi_x[epi_max_index]
        fid_infill = all_epi_fid[epi_max_index]

        return x_infill, fid_infill


    def minimize_with_deap(self, epi_func, bounds, num_levels,
                           pop_size=50, ngen=40, seed=None):
        """
        Minimize an acquisition function jointly over x and fidelity level using DEAP.

        :param epi_func: Callable function to minimize, of the form ``epi_func(x, fid)``.
        :type epi_func: callable

        :param bounds: Bounds for the continuous design variables.
        :type bounds: list[tuple[float, float]]

        :param num_levels: Number of discrete fidelity levels (0 .. num_levels - 1).
        :type num_levels: int

        :param pop_size: Population size.
        :type pop_size: int

        :param ngen: Number of generations.
        :type ngen: int

        :param seed: Random seed for reproducibility.
        :type seed: int, optional

        :return: A tuple ``(best_x, best_fid)`` where:
                 - ``best_x`` is the best variable found
                 - ``best_fig`` is the corresponding fidelity level
        :rtype: tuple[np.ndarray, int]
        """

        num_dim = self.bounds.shape[0]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # define DEAP classes (guard against re-creation if function is called multiple times)
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # initialization for x variables
        for i in range(self.num_dim):
            toolbox.register(f"attr_x_{i}", random.uniform, bounds[i, 0], bounds[i, 1])

        # initialization for fidelity variable (int)
        toolbox.register("attr_fid", random.randint, 0, num_levels-1)

        # combine x variables + fidelity
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [toolbox.__getattribute__(f"attr_x_{i}") for i in range(num_dim)] + [toolbox.attr_fid],
            n=1,
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # evaluation
        def evaluate(individual):
            x = np.clip(np.array(individual[:-1]), bounds[:, 0], bounds[:, 1])

            fid = int(np.clip(round(individual[-1]), 0, num_levels-1))

            return (-epi_func(x, fid),)

        toolbox.register("evaluate", evaluate)

        # operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)

        # continuous mutation for x, integer mutation for fid
        def mutate(individual, indpb=0.2):
            for i in range(num_dim):
                if random.random() < indpb:
                    individual[i] = random.uniform(bounds[i, 0], bounds[i, 1])

            # mutate fidelity with small probability
            if random.random() < indpb:
                individual[-1] = random.randint(0, num_levels - 1)

            # clip variables within the boundaries
            for i in range(num_dim):
                individual[i] = np.clip(individual[i], bounds[i, 0], bounds[i, 1])

            # clip the fidelity with the problem's levels
            individual[-1] = np.clip(individual[-1], 0, num_levels-1)

            return (individual,)

        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # create population
        pop = toolbox.population(n=pop_size)

        # run evolution
        algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=ngen,
                            stats=None, halloffame=None, verbose=False)

        # get best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_x = np.array(best_ind[:-1])
        best_fid = int(round(best_ind[-1]))
        best_f = evaluate(best_ind)[0]

        return best_x, best_fid
