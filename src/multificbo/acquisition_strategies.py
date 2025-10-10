import numpy as np
import scipy.stats as stats
import scipy.optimize as so

from multificbo.surrogate_models import Surrogate
from multificbo.acquisition_functions import expected_improvement, probability_of_improvement
from multificbo.acquisition_functions import log_ei

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

        obj_surrogate = optimizer.obj_surrogate
        cstr_surrogates = optimizer.cstr_surrogates

        if optimizer.num_cstr > 0:
            constrained = True
        else:
            constrained = False

        def scipy_ei_wrapper(x):

            x = x.reshape(1, -1)

            mu = obj_surrogate.predict_value(x)
            s2 = obj_surrogate.predict_variance(x)

            return -self.acq_func(mu, s2, f_min)

        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        acq_res_x = np.empty_like(acq_x0)
        acq_res_f = np.empty(acq_multistart)

        scipy_cstr = []
        if constrained:
            scipy_cstr = [{"type": "ineq", "fun": lambda x, c_gp=c_gp: -c_gp.predict_value(x.reshape(1, -1))} for c_gp in
                          cstr_surrogates]

        for i in range(acq_x0.shape[0]):
            so_x0 = acq_x0[i, :]
            acq_res = so.minimize(scipy_ei_wrapper, so_x0, bounds=bounds, method="COBYLA", constraints=scipy_cstr, tol=1e-4)
            acq_res_x[i, :] = acq_res.x
            acq_res_f[i] = acq_res.fun

        # TODO: check if solution respect the constraint

        next_index = np.argmin(acq_res_f)
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

                mu = obj_surrogate.predict_value(x)
                s2 = obj_surrogate.predict_variance(x)

                return -self.acq_func(mu, s2, f_min) * self.alpha1(x, lvl, n_level - 1) * alpha3

            cstr_surrogates = optimizer.cstr_surrogates
            constrained = optimizer.constrained

            if cstr_surrogates is None or cstr_surrogates == []:
                constrained = True

            scipy_cstr = []
            if constrained:
                scipy_cstr = [{"type": "ineq", "fun": lambda x, c_gp=c_gp: -c_gp.predict_value(x.reshape(1, -1))} for
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

        if self.optimizer is not None:
            self.mfck = self.optimizer.obj_surrogate.mfck
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

        means, vars = self.mfck.predict_all_levels(x)
        cov = self.mfck.predict_level_covariances(x, level)

        pi = probability_of_improvement(means[-1].reshape(-1, 1), vars[-1].reshape(-1, 1), self.f_min)

        corr = self.fidelity_correlation(cov, vars[level].reshape(-1, 1), vars[-1].reshape(-1, 1))

        # TODO: add cost ratio here

        density = self.sample_density(x, level, self.mfck)

        return pi * corr * density

    def execute_infill_strategy(self, optimizer) -> list:

        self.current_level = 0
        self.f_min, _ = self.predicted_f_min(-1)

        def epi_scipy_wrapper(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            epi = self.epi(x, self.current_level).ravel()
            cost_ratio = (optimizer.costs[-1]/optimizer.costs[self.current_level])

            satisfying = 1.0

            # TODO: test implementation
            for cstr in range(len(optimizer.cstr_surrogates)):
                g_pred = optimizer.cstr_surrogates[cstr].predict_value(x)
                s2_pred = optimizer.cstr_surrogates[cstr].predict_variance(x)

                satisfying *= stats.norm.cdf(g_pred/np.sqrt(s2_pred))

            return -epi * cost_ratio * satisfying


        dim = optimizer.num_dim
        bounds = optimizer.domain
        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        all_epi_max = []
        all_epi_argmax = []
        all_epi_fid = []

        for lvl in range(optimizer.num_levels):
            self.current_level = lvl

            for i in range(acq_x0.shape[0]):

                res = so.minimize(epi_scipy_wrapper,
                                  acq_x0[i, :],
                                  method="COBYLA",
                                  bounds=bounds,
                                  tol=1e-4)

                all_epi_max.append(res.fun)
                all_epi_argmax.append(res.x)
                all_epi_fid.append(self.current_level)


        epi_max_index = np.argmin(all_epi_max)
        x_infill = all_epi_argmax[epi_max_index]
        fid_infill = all_epi_fid[epi_max_index]

        infill = []

        for lvl in range(optimizer.num_levels):
            if lvl == fid_infill:
                infill.append(x_infill)
            else:
                infill.append(None)

        return infill





