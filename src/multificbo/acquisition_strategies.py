import numpy as np
import scipy.stats as stats
import scipy.optimize as so
from warnings import warn
import random

from abc import ABC, abstractmethod

from deap import base, creator, tools, algorithms

# from multificbo.optimizer import Optimizer
from multificbo.surrogate_models import Surrogate, SmtMFK
from multificbo.acquisition_functions import expected_improvement, log_ei, probability_of_improvement, fidelity_correlation

from multificbo.suboptimizers.orthomads import orthomads


# TODO: acquisition strategy class template

class AcquisitionStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    @abstractmethod
    def execute_infill_strategy(self, optimizer):
        raise Exception("Acquisition Strategy not implemented.")


class MonoFiAcqStrat(AcquisitionStrategy):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__()

        self.optimizer = optimizer
        self.acq_func = kwargs.pop("acq_func", log_ei)
        self.sub_optimizer = kwargs.pop("sub_optimizer", "COBYLA")
        self.n_start = kwargs.pop("n_start", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if self.optimizer is not None and self.n_start is None:
            self.n_start = 10*optimizer.num_dim

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    def execute_infill_strategy(self, optimizer) -> np.ndarray:

        if self.sub_optimizer == "ORTHOMADS":
            return self.acq_strategy_orthomads(optimizer)
        elif self.sub_optimizer == "COBYLA":
            return self.acq_strategy_cobyla(optimizer)
        else:
            raise Exception(f"{self.sub_optimizer} is not a valide sub optimizer.")


    def acq_strategy_orthomads(self, optimizer) -> np.ndarray:

        def ortho_wrapper(x: np.ndarray) -> np.ndarray:
            mu = optimizer.obj_surrogate.predict_values(x)
            s2 = optimizer.obj_surrogate.predict_variances(x)
            return -self.acq_func(mu, s2, optimizer.f_min_scaled).ravel()

        sampler = stats.qmc.LatinHypercube(d=optimizer.num_dim)
        x0_multistart = sampler.random(self.n_start)
        x0_multistart = stats.qmc.scale(x0_multistart, optimizer.domain[:, 0], optimizer.domain[:, 1])

        fmin = []
        xmin = []
        cmin = []
        feasible_counter = 0

        constraints = []
        for c_surrogate in optimizer.cstr_surrogates:
            constraints.append(
                c_surrogate.predict_values
            )

        # time = 0
        for i in range(x0_multistart.shape[0]):
            x0 = x0_multistart[i, :]

            # t0 = perf_counter()
            res = orthomads(ortho_wrapper, x0, bounds=optimizer.domain, constraints=constraints,
                            max_iter=1_000, verbose=False)
            # t1 = perf_counter()
            # time += (t1 - t0)

            print(f"iter={res.num_iter} | fmin={res.fun} |")

            cstr_values = np.full(optimizer.num_cstr, np.inf)
            for c_id, c_surrogate in enumerate(optimizer.cstr_surrogates):
                cstr_values[c_id] = c_surrogate.predict_values(res.x.reshape(1, -1)).item()

            if np.all(cstr_values <= 0):
                feasible_counter += 1
                fmin.append(res.fun)
                xmin.append(res.x)

        index = np.array(fmin).argmin()
        next_x = np.array(xmin)[index, :]

        return next_x

    def acq_strategy_cobyla(self, optimizer) -> np.ndarray:
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

        f_min = optimizer.f_min_scaled
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
        if num_cstr > 0:
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
        if num_cstr == 0:
            feas_mask = np.full_like(acq_res_f, True)
        else:
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
    def __init__(self, optimizer=None, **kwargs):
        super().__init__()

        self.optimizer = optimizer
        self.acq_func = kwargs.pop("acq_func", log_ei)
        self.sub_optimizer = kwargs.pop("sub_optimizer", "COBYLA")
        self.n_start = kwargs.pop("n_start", None)

        # possible criteria : obj-only, optimistic, pessimistic, average, cstr-only (only 1 cstr)
        self.fidelity_crit = kwargs.pop("fidelity_crit", "obj-only")

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if self.optimizer is not None and self.n_start is None:
            self.n_start = 10*optimizer.num_dim

        self.mono_fi_strat = MonoFiAcqStrat(optimizer=optimizer,
                                            acq_func=self.acq_func,
                                            sub_optimizer=self.sub_optimizer,
                                            n_start=self.n_start,
                                            )

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")


    def execute_infill_strategy(self, optimizer) -> list:
        if optimizer.num_levels == 1:
            raise Exception("Incorrect acquisition strategy. Problem is not multi-fidelity.")

        return self.mf_acq_strategy(optimizer)


    def compute_sigma2_red(self, x_pred: np.ndarray, surrogate: SmtMFK) -> np.ndarray:

        # np.ndarray(num_points, num_levels), list[np.ndarray(num_points)]
        s2, rho2 = surrogate.mfk.predict_variances_all_levels(x_pred)
        num_levels = s2.shape[1]

        tot_rho2 = np.ones((x_pred.shape[0], num_levels))
        s2_red = np.empty((x_pred.shape[0], num_levels))

        for k in range(num_levels):
            for l in range(k, num_levels-1):
                tot_rho2[:, k] *= rho2[l][:]

            s2_red[:, k] = s2[:, k] * tot_rho2[:, k]

        # np.array(num_points, num_levels)
        return s2_red


    def compute_norm_squared_cost(self, costs: list[float]) -> np.ndarray:

        num_levels = len(costs)
        tot_costs2 = np.empty(num_levels)

        for k in range(num_levels):
            tot_costs2[k] = np.sum(costs[0:k+1])**2

        # normalize the aggregate costs squared by its maximum
        tot_costs2 /= np.max(tot_costs2)

        return tot_costs2

    def compute_norm_sigma2_red(self, x_pred: np.ndarray, norm_costs2: list[float], surrogate: SmtMFK) -> np.ndarray:

        num_levels = len(norm_costs2)

        s2_red = self.compute_sigma2_red(x_pred, surrogate)
        s2_norm = np.empty_like(s2_red)

        for k in range(num_levels):
            s2_norm[:, k] = s2_red[:, k] / norm_costs2[k]

        return s2_norm


    def compute_all_s2_red_norm(self, x_pred: np.ndarray, costs: list[float], surrogates: list[SmtMFK]) -> list[np.ndarray]:

        num_pts = x_pred.shape[0]
        num_levels = len(costs)

        norm_costs2 = self.compute_norm_squared_cost(costs)

        s2_red_norm = [np.empty((num_pts, num_levels)) for _ in range(len(surrogates))]

        for i, surrogate in enumerate(surrogates):
            s2_red_norm[i] = self.compute_norm_sigma2_red(x_pred, norm_costs2, surrogate)

        return s2_red_norm


    def select_fidelity_level(self, x_pred: np.ndarray, costs: list[float], all_surrogates: list[SmtMFK], criterion: str) -> np.ndarray:

        num_pts: int = x_pred.shape[0]
        # level: np.ndarray = np.zeros(num_pts)

        if criterion == "obj-only":
            surrogates = [all_surrogates[0]]
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, surrogates)

            level = s2_red_norm[0].argmax(axis=1)

        elif criterion == "optimistic":
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)

            # TODO: make it compatible with multiple infill points
            level = s2_red_norm[0].argmax(axis=1)

            for i in range(1, len(all_surrogates)):
                level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).min(axis=0)

        elif criterion == "pessimistic":
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)

            level = s2_red_norm[0].argmax(axis=1)

            for i in range(1, len(all_surrogates)):
                level = np.vstack((level, s2_red_norm[i].argmax(axis=1))).max(axis=0)

        elif criterion == "average":
            # s2_red of each surrogate is normalized by the cost. Should it be normalized after the sum?
            # -> should be the same
            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, all_surrogates)
            s2_red_avg = np.zeros((num_pts, s2_red_norm[0].shape[1]))

            # sum the s2_red from all surrogates
            for i in range(len(all_surrogates)):
                s2_red_avg[:, :] += s2_red_norm[i][:, :]

            level = s2_red_avg.argmax(axis=1)

        elif criterion == "cstr-only":

            if len(all_surrogates) == 1:
                raise Exeption("cstr-only criterion requires one constraint surrogate.")

            surrogates = all_surrogates[1:]

            if len(surrogates) > 1:
                raise Exception("cstr-only is not implemented for more than 1 constraints.")

            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, surrogates)

            level = s2_red_norm[0].argmax(axis=1)

        # np.ndarray(num_pts) -> fidelity level for each infill points
        return level


    def mf_acq_strategy(self, optimizer) -> list[np.ndarray]:

        # set acquisition function to mono-fidelity strategy
        self.mono_fi_strat.acq_func = self.acq_func

        # get next sampling point
        next_x: list[np.ndarray] = self.mono_fi_strat.execute_infill_strategy(optimizer)

        num_levels: int = optimizer.num_levels

        obj_surrogate: Surrogate = optimizer.obj_surrogate
        cstr_surrogates : list[Surrogate] = optimizer.cstr_surrogates
        obj_cost: list[float] = optimizer.costs

        all_surrogates: list[Surrogate] = [obj_surrogate]
        for c_surrogate in cstr_surrogates:
            all_surrogates.append(c_surrogate)

        level: int = self.select_fidelity_level(next_x.reshape(1, -1),
                                                obj_cost,
                                                all_surrogates,
                                                self.fidelity_crit)


        # optimizer.iter_data["s2_red"] = s2_red
        # optimizer.iter_data["tot_cost2"] = tot_cost
        # optimizer.iter_data["norm_tot_cost2"] = norm_tot_cost
        # optimizer.iter_data["fid_crit"] = fid_crit

        # max_level = np.argmax(fid_crit)

        optimizer.iter_data["infill_x"] = next_x
        optimizer.iter_data["infill_max_level"] = level    # redundant

        # for each index    -> None:            do not sample level
        #                   -> np.ndarray():    sample level at said point
        # warning -> does not work if next_x is more than 1 point
        next_x_all_lvl: list[np.ndarray] = [None] * num_levels
        for k in range(num_levels+1):
            if k <= level:
                next_x_all_lvl[k] = next_x

        return next_x_all_lvl

# class MFEI:
#     def __init__(self, acq_func=expected_improvement, optimizer=None):
#         super().__init__()
#
#         self.optimizer = optimizer
#         self.acq_func = acq_func
#         self.n_start = 10
#
#     def alpha1(self, x: np.ndarray, l: int, m:int ):
#
#         # or approximate a pearson correlation factor?
#
#         mfk = self.optimizer.obj_surrogate.mfk
#
#         s2_pred, rho2 = mfk.predict_variances_all_levels(x)
#
#         var_l = s2_pred[:, l]
#         var_m = s2_pred[:, m]
#
#         kappa = var_l
#
#         for ll in range(l, m):
#             kappa *= np.sqrt(rho2[ll])
#
#         alpha1 = kappa/np.sqrt(var_l * var_m)
#         # print(f"alpha1 = {alpha1}")
#         #
#         # mu_l = mfk._predict_intermediate_values(x, l+1)
#         # print(f"mu_l = {mu_l}")
#         # mu_mm = mfk._predict_intermediate_values(x, m+1)
#         # print(f"mu_mm = {mu_mm}")
#         # mu_m = mfk._predict_values(x)
#         # print(f"mu_m = {mu_m}")
#         #
#         # samples_l = np.random.normal(loc=mu_l.squeeze(), scale=np.sqrt(var_l.squeeze()), size=10_000)
#         # samples_m = np.random.normal(loc=mu_m.squeeze(), scale=np.sqrt(var_m.squeeze()), size=10_000)
#         #
#         # print(f"pearson = \n{np.corrcoef(samples_l, samples_m)}")
#         #
#         #
#         # print(f"corr = {np.sqrt(var_l.squeeze())/np.abs(mu_m - mu_l) + np.sqrt(var_m)}")
#         # apply correction if correlation coefficient is > 1. Should raise warning?
#         alpha1 = np.where(alpha1 > 1, 1, alpha1)
#         return alpha1
#
#     def alpha2(self, x: np.ndarray, l: int):
#         # assume no noise
#         return 1.0
#
#     def alpha3(self, l):
#
#         costs = self.optimizer.obj_surrogate.cost
#
#         tot_cost_m = np.sum(costs)
#         tot_cost_l = np.sum(costs[0:l+1])
#
#         return tot_cost_m/tot_cost_l
#
#
#     def execute_infill_strategy(self, optimizer) -> list:
#
#         obj_surrogate = optimizer.obj_surrogate
#         dim = optimizer.num_dim
#         n_level = optimizer.num_levels
#         f_min = optimizer.f_min_scaled
#         bounds = optimizer.domain
#
#         acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better
#
#         acq_x0 = acq_sampler.random(self.n_start)
#         acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])
#
#         best_x_per_level = []
#         best_f_per_level = []
#
#         for lvl in range(n_level):
#
#             alpha3 = self.alpha3(lvl)
#
#             acq_res_x = np.empty_like(acq_x0)
#             acq_res_f = np.empty(self.n_start)
#
#             def scipy_ei_wrapper(x):
#                 x = x.reshape(1, -1)
#
#                 mu = obj_surrogate.predict_values(x)
#                 s2 = obj_surrogate.predict_variances(x)
#
#                 return -self.acq_func(mu, s2, f_min) * self.alpha1(x, lvl, n_level - 1) * alpha3
#
#             cstr_surrogates = optimizer.cstr_surrogates
#             constrained = optimizer.constrained
#
#             if cstr_surrogates is None or cstr_surrogates == []:
#                 constrained = True
#
#             scipy_cstr = []
#             if constrained:
#                 scipy_cstr = [{"type": "ineq", "fun": lambda x, c_gp=c_gp: -c_gp.predict_values(x.reshape(1, -1))} for
#                               c_gp in
#                               cstr_surrogates]
#
#             for i in range(acq_x0.shape[0]):
#
#                 so_x0 = acq_x0[i, :]
#                 acq_res = so.minimize(scipy_ei_wrapper, so_x0,
#                                       bounds=bounds,
#                                       method="COBYLA",
#                                       constraints=scipy_cstr,
#                                       tol=1e-4)
#
#                 acq_res_x[i, :] = acq_res.x
#                 acq_res_f[i] = acq_res.fun
#
#             index = np.argmin(acq_res_f)
#             best_x_per_level.append( acq_res_x[index, :] )
#             best_f_per_level.append( acq_res_f[index] )
#
#             optimizer.iter_data[f"lvl{lvl}_acq_f"] = acq_res_f[index]
#             optimizer.iter_data[f"lvl{lvl}_alpha1"] = self.alpha1(np.array([acq_res_x[index, :]]), lvl, n_level-1)
#             optimizer.iter_data[f"lvl{lvl}_alpha3"] = alpha3
#
#         next_level = np.argmax(best_f_per_level)
#         next_x = best_x_per_level[next_level]
#
#         optimizer.iter_data["infill_x"] = next_x
#         optimizer.iter_data["infill_max_level"] = next_level
#
#         next_x_all_lvl = [None] * n_level
#
#         for k in range(next_level+1):
#             if k <= next_level:
#                 next_x_all_lvl[k] = next_x
#
#         return next_x_all_lvl

class MFEI(AcquisitionStrategy):

    def __init__(self, acq_func=expected_improvement, optimizer=None):

        self.optimizer = optimizer
        self.acq_func = acq_func
        self.n_start = 10

        self.num_dim = 0
        self.num_cstr = 0
        self.num_levels = 0
        self.bounds = None

        self.obj_surrogate = None
        self.cstr_surrogates = []

        self.sub_optimizer = "COBYLA"

        if self.optimizer is not None:
            self.num_dim = self.optimizer.num_dim
            self.num_cstr = self.optimizer.num_cstr
            self.num_levels = self.optimizer.num_levels
            self.bounds = self.optimizer.domain

            self.obj_surrogate = self.optimizer.obj_surrogate
            self.cstr_surrogates = self.optimizer.cstr_surrogates

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")


    def augmented_ei(self, x, level):

        if x.ndim == 1:
            x = x.reshape(1, -1)

        means, vars = self.obj_surrogate.mfck.predict_all_levels(x)
        cov = self.obj_surrogate.mfck.predict_level_covariances(x, level)

        ei = self.acq_func(means[-1].reshape(-1, 1), vars[-1].reshape(-1, 1), self.f_min)
        alpha1 = fidelity_correlation(cov, vars[level].reshape(-1, 1), vars[-1].reshape(-1, 1))
        alpha3 = self.optimizer.costs[-1]/self.optimizer.costs[level]

        return ei * alpha1 * alpha3

    def minimize_with_scipy(self) -> tuple:

        self.current_level = 0

        def augmented_ei_scipy_wrapper(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            augmented_ei = self.augmented_ei(x, self.current_level).ravel()
            return -augmented_ei

        dim = self.optimizer.num_dim
        bounds = self.optimizer.domain
        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=self.num_dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, self.bounds[:, 0], self.bounds[:, 1])

        acq_res_x = np.empty((self.n_start*self.num_levels, self.num_dim))
        acq_res_f = np.empty(self.n_start*self.num_levels)
        acq_res_lvl = np.empty(self.n_start*self.num_levels)

        if self.num_cstr > 0:
            acq_res_c = np.empty((self.n_start * self.num_levels, self.num_cstr))

        scipy_cstr = []
        for c_surrogate in self.cstr_surrogates:
            scipy_cstr.append(
                {"type": "ineq",
                 "fun": lambda x, f=c_surrogate.mfck.predict_all_levels: -f(x.reshape(1, -1))[0][-1].ravel()}
            )

        for lvl in range(self.num_levels):
            self.current_level = lvl

            for i in range(acq_x0.shape[0]):

                j = lvl*self.n_start + i

                res = so.minimize(augmented_ei_scipy_wrapper,
                                  acq_x0[i, :],
                                  method="COBYLA",
                                  bounds=bounds,
                                  constraints=scipy_cstr,
                                  tol=1e-4)

                acq_res_x[j, :] = res.x
                acq_res_f[j] = res.fun
                acq_res_lvl[j] = self.current_level

                for c_id, c_surrogate in enumerate(self.cstr_surrogates):
                    acq_res_c[j, c_id] = c_surrogate.predict_values(acq_res_x[j, :].reshape(1, -1)).ravel()

        # check if solution respect the constraints
        if self.optimizer.num_cstr == 0:
            feas_mask = np.full_like(acq_res_f, True)
        else:
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

        x_infill = acq_res_x[next_index, :]
        lvl_infill = acq_res_lvl[next_index]

        return x_infill, lvl_infill


    def execute_infill_strategy(self, optimizer) -> list:

        self.optimizer = optimizer
        self.f_min = self.optimizer.f_min_scaled

        x_infill, fid_infill = self.minimize_with_scipy()

        infill = []
        for lvl in range(self.num_levels):
            if lvl == fid_infill:
                infill.append(x_infill)
            else:
                infill.append(None)

        return infill



class VFPI(AcquisitionStrategy):

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

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

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
