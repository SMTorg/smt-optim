import numpy as np
import scipy.stats as stats
import scipy.optimize as so
from warnings import warn
import random
from time import perf_counter
from functools import partial

from abc import ABC, abstractmethod

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

class MFSEGO_EQ(AcquisitionStrategy):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__()

        self.optimizer = optimizer
        self.acq_func = kwargs.pop("acq_func", log_ei)                  # expected_improvement, log_ei
        self.fmin_crit = kwargs.pop("fmin_crit", "min_rscv")            # min_rscv, fmin, mean_rscv
        # self.sub_optimizer = kwargs.pop("sub_optimizer", "COBYLA")
        self.n_start = kwargs.pop("n_start", None)
        self.fidelity_crit = kwargs.pop("fidelity_crit", "obj-only")    # obj-only, average, optimistic, pessimistic
        self.select_fidelity = kwargs.pop("select_fidelity", True)
        self.min_rscv_first = kwargs.pop("min_rscv_first", False)
        self.filter_rscv = kwargs.pop("filter_rscv", False)
        self.optimize_best = kwargs.pop("optimize_best", False)
        self.relax_constraints = kwargs.pop("relax_constraints", False)
        self.cr_override = kwargs.pop("cr_override", None)                  # override optimizer Cost Ratio

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if self.optimizer is not None and self.n_start is None:
            self.n_start = 10*optimizer.num_dim

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    def execute_infill_strategy(self, optimizer) -> list[np.ndarray]:

        acq_data = {}

        self.fmin = self.get_fmin(optimizer, fmin_crit=self.fmin_crit)
        acq_data["fmin"] = self.fmin
        acq_data["fmin_descaled"] = self.fmin * optimizer.yt[-1].std(axis=0) + optimizer.yt[-1].mean()

        # generate starting points for the multistart optimization
        gen_t0 = perf_counter()
        multi_x0 = self.generate_multistart_points(optimizer)
        gen_t1 = perf_counter()
        acq_data["generate_init_points_time"] = gen_t1 - gen_t0

        multi_x = np.empty_like(multi_x0)
        multi_f = np.empty(multi_x0.shape[0])
        multi_c = np.empty((multi_x0.shape[0], optimizer.num_cstr))
        multi_success = np.full(multi_x0.shape[0], False)

        # scipy objective wrapper
        def scipy_acq_func(x):
            x = x.reshape(1, -1)
            mu = optimizer.obj_surrogate.predict_values(x)
            s2 = optimizer.obj_surrogate.predict_variances(x)
            return -self.acq_func(mu, s2, self.fmin).item()

        # scipy constraint wrapper (for scipy, the feasible domain is g >= 0)
        scipy_cstr = []
        for c_id, c_config in enumerate(optimizer.cstr_config):

            if c_config.type in ["less", "greater"]:
                c_type = "ineq"
            elif c_config.type == "equal":
                c_type = "eq"
            else:
                raise Exception(f"Unexpected constraint type: {c_config.type}")

            def wrap_constraint(mu_func, s2_func, type, relax=False):

                def sp_constraint(x):
                    x = x.reshape(1, -1)
                    mu = mu_func(x).item()

                    if relax:
                        s = np.sqrt(s2_func(x).item())
                        if type == "ineq":
                            return -(mu - 3*s)
                        elif type == "eq":
                            return -(np.abs(mu) - 3*s)

                    return -mu

                return sp_constraint


            mu_func = optimizer.cstr_surrogates[c_id].predict_values
            s2_func = optimizer.cstr_surrogates[c_id].predict_variances
            sp_cstr_func = wrap_constraint(mu_func, s2_func, c_type, relax=self.relax_constraints)

            if self.relax_constraints:
                c_type = "ineq"

            scipy_cstr.append(
                {
                    "type": c_type,
                    "fun": lambda x, f=sp_cstr_func: f(x),
                }
            )


        for i in range(multi_x0.shape[0]):

            # for cstr in scipy_cstr:
            #     print(f"ini c value = {-cstr["fun"](multi_x0[i, :])}")

            res = so.minimize(scipy_acq_func,
                              x0=multi_x0[i, :],
                              bounds=optimizer.domain_scaled,
                              constraints=scipy_cstr,
                              method="SLSQP",
                              tol=1e-8,
                              options={"maxiter": 200*optimizer.num_dim})


            # apply L1 correction for the bounds
            # x_corrected = np.where(res.x < optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 0], res.x)
            # x_corrected = np.where(x_corrected > optimizer.domain_scaled[:, 1], optimizer.domain_scaled[:, 1], res.x)

            x_clipped = np.clip(res.x, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])

            multi_x[i, :] = x_clipped
            multi_f[i] = -scipy_acq_func(x_clipped)

            if optimizer.num_cstr > 0:
                for c_id in range(len(scipy_cstr)):
                    multi_c[i, c_id] = -scipy_cstr[c_id]["fun"](x_clipped)

            multi_success[i] = res.success

            # print(f"{i+1}/{self.n_start} | f={multi_f[i]} | Delta_x={np.linalg.norm(multi_x0[i, :] - x_corrected)} | c={multi_c[i, :]} | {res.success} | {res.message} | time={t1-t0:.3f}")

        if optimizer.num_cstr == 0:
            feas_mask = np.full(multi_x0.shape[0], True)
        else:

            rscv = self.compute_rscv(multi_c, optimizer.cstr_config, g_tol=0., h_tol=0.)

            if self.filter_rscv:
                feas_mask = np.where(rscv <= 1e-4, True, False)
            else:
                feas_mask = np.full(multi_x0.shape[0], True)

        if np.any(feas_mask):
            idx = np.argmax(np.where(feas_mask, multi_f, np.inf))
        else:
            idx = np.argmax(rscv)

        success_rate = np.count_nonzero(multi_success)/multi_success.shape[0]

        # f_min: float = multi_f[idx]
        x_min: np.ndarray = multi_x[idx, :]

        acq_data["multi_idx"] = idx
        acq_data["multi_x"] = multi_x
        acq_data["multi_f"] = multi_f
        acq_data["multi_c"] = multi_c
        acq_data["acq_success"] = success_rate

        if optimizer.num_cstr > 0:
            acq_data["rscv"] = rscv

        if self.optimize_best:

            if optimizer.num_cstr > 0:
                optimized_cstr = np.empty(optimizer.num_cstr)

            res = so.minimize(scipy_acq_func,
                              x0=x_min,
                              bounds=optimizer.domain_scaled,
                              constraints=scipy_cstr,
                              method="SLSQP",
                              tol=1e-15,
                              options={"maxiter": 50 * optimizer.num_dim})


            optimized_x = np.clip(res.x, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])
            acq_data["optimized_x"] = optimized_x

            optimized_l2 = np.linalg.norm(optimized_x - x_min)
            acq_data["optimized_l2"] = optimized_l2

            optimized_acq = scipy_acq_func(optimized_x)
            acq_data["optimized_acq"] = optimized_acq

            if optimizer.num_cstr > 0:
                for c_id in range(len(scipy_cstr)):
                    optimized_cstr[c_id] = -scipy_cstr[c_id]["fun"](optimized_x)
                    acq_data["optimized_cstr"] = optimized_cstr

            # update x_min
            x_min = optimized_x

        # select highest fidelity level to sample
        fid_crit_t0 = perf_counter()
        if optimizer.num_levels > 1 and self.select_fidelity:

            all_surrogates = [optimizer.obj_surrogate]
            for c_surrogate in optimizer.cstr_surrogates:
                all_surrogates.append(c_surrogate)

            if self.cr_override is not None:
                costs = self.cr_override
            else:
                costs = optimizer.costs

            levels, s2_red_norm = self.select_fidelity_level(x_min.reshape(1, -1),
                                                             costs,
                                                             all_surrogates,
                                                             self.fidelity_crit)
            level = levels.item()

        else:
            level = optimizer.num_levels-1

        acq_data["infill_level"] = level

        if optimizer.num_levels > 1 and self.select_fidelity:
            acq_data["normalized_s2_reduction"] = s2_red_norm

        next_x = []
        for lvl in range(optimizer.num_levels):
            if lvl <= level:
                next_x.append(x_min.copy())
            else:
                next_x.append(None)
        fid_crit_t1 = perf_counter()
        acq_data["fid_crit_time"] = fid_crit_t1 - fid_crit_t0

        # log expected values
        expected_values = np.empty(optimizer.num_cstr+1)
        expected_values[0] = optimizer.obj_surrogate.predict_values(x_min.reshape(1, -1)).item()

        if optimizer.scaling:
            yt_scaled, yt_mean, yt_std = optimizer._standardize_data(optimizer.yt[-1])
            expected_values[0] *= yt_std
            expected_values[0] +=  yt_mean

        for c_id, c_surrogate in enumerate(optimizer.cstr_surrogates):
            expected_values[c_id+1] = c_surrogate.predict_values(x_min.reshape(1, -1)).item()
            if optimizer.scaling:
                yt_scaled, yt_mean, yt_std = optimizer._standardize_data(optimizer.ct[-1][:, c_id])
                expected_values[c_id+1] *= yt_std

        acq_data["expected_values"] = expected_values

        optimizer.iter_data["acquisition"] = acq_data

        return next_x


    def get_fmin(self, optimizer, rscv_tol: float = 0.0, fmin_crit: str = "min_rscv") -> float:

        # no constraint
        if optimizer.num_cstr == 0 or fmin_crit == "fmin":
            idx = optimizer.yt_scaled[-1].argmin()

        elif fmin_crit == "min_rscv":
            ct_rscv = self.compute_rscv(optimizer.ct_scaled[-1], optimizer.cstr_config)
            feas_mask = np.where(ct_rscv <= rscv_tol, True, False)

            if np.any(feas_mask):
                idx = np.argmin(np.where(feas_mask, optimizer.yt_scaled[-1][:, 0], np.inf))
            else:
                idx = np.argmin(ct_rscv)

        elif fmin_crit == "mean_rscv":
            rscv = self.compute_rscv(optimizer.yt_scaled[-1], optimizer.cstr_config)
            mean_rscv = rscv.mean()

            feas_mask = np.where(rscv <= mean_rscv, True, False)
            idx = np.argmin(np.where(feas_mask, optimizer.yt_scaled[-1][:, 0], np.inf))

        else:
            raise Exception(f"{fmin_crit} is not a valid fmin_crit")

        fmin = optimizer.yt_scaled[-1][idx, 0]

        return fmin


    def generate_multistart_points(self, optimizer) -> np.ndarray:

        sampler = stats.qmc.LatinHypercube(d=optimizer.domain_scaled.shape[0])

        # LHS filter
        large_x0 = sampler.random(10*self.n_start)
        # large_x0 = sampler.random(self.n_start)
        large_x0 = stats.qmc.scale(large_x0, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])

        mu = optimizer.obj_surrogate.predict_values(large_x0)
        s2 = optimizer.obj_surrogate.predict_variances(large_x0)
        large_f = -self.acq_func(mu, s2, self.fmin)

        # no constraints -> selects the best starting points
        if optimizer.num_cstr == 0:
            sorted_idx = np.argsort(large_f.ravel())

        # with constraints -> selects the best points with the lowest constraint violation
        else:
            large_c = np.empty((large_x0.shape[0], optimizer.num_cstr))

            for c_id, c_surrogate in enumerate(optimizer.cstr_surrogates):
                large_c[:, c_id] = c_surrogate.predict_values(large_x0).ravel()

            rscv = self.compute_rscv(large_c, optimizer.cstr_config)
            sorted_idx = np.lexsort((large_f.ravel(), rscv))
            rscv = rscv[sorted_idx][:self.n_start]

        multi_x0 = large_x0[sorted_idx][:self.n_start, :]

        # with constraints -> try to reduce the starting point RSCV
        if optimizer.num_cstr > 0 and self.min_rscv_first:

            def min_rscv(x):
                cstr_values = np.empty((1, len(optimizer.cstr_surrogates)))
                for c_id, c_surrogate in enumerate(optimizer.cstr_surrogates):
                    cstr_values[0, c_id] = c_surrogate.predict_values(x.reshape(1, -1)).item()
                return self.compute_rscv(cstr_values, optimizer.cstr_config).item()


            for i in range(multi_x0.shape[0]):
                # try to reduce the constraint violation if the starting point is not feasible
                if rscv[i] != 0.0:
                    res = so.minimize(min_rscv,
                                      x0=multi_x0[i, :],
                                      bounds=optimizer.domain_scaled,
                                      method="COBYLA",
                                      tol=1e-8,
                                      options={"maxiter": 50*optimizer.num_dim})

                    rscv[i] = res.fun
                    multi_x0[i, :] = res.x

        return multi_x0


    def compute_rscv(self, cstr_array: np.ndarray, cstr_config: list, g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:

        scv = np.full_like(cstr_array, 0.0)     # Square Constraint Violation

        for c_id, c_config in enumerate(cstr_config):

            if c_config.type in ["less", "greater"]:
                valid_mask = cstr_array[:, c_id] <= g_tol
                scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

            elif c_config.type == "equal":
                valid_mask = np.abs(cstr_array[:, c_id]) <= h_tol
                scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

            else:
                raise Exception(f"{c_config.type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'.")

        rscv = np.sqrt(scv.sum(axis=1))

        return rscv

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
                raise Exception("cstr-only criterion requires one constraint surrogate.")

            surrogates = all_surrogates[1:]

            if len(surrogates) > 1:
                raise Exception("cstr-only is not implemented for more than 1 constraints.")

            s2_red_norm = self.compute_all_s2_red_norm(x_pred, costs, surrogates)

            level = s2_red_norm[0].argmax(axis=1)

        # np.ndarray(num_pts) -> fidelity level for each infill points
        return level, s2_red_norm



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
        return level, s2_red_norm


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
        self.n_start = 50 # 10

        self.f_min = np.inf

        if self.optimizer is not None:
            self.mfck = self.optimizer.obj_surrogate.mfck
            self.n_start *= self.optimizer.num_dim
        else:
            self.mfck = None

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    def predicted_f_min(self, level: int) -> tuple[float, np.ndarray]:

        dim = self.optimizer.num_dim
        bounds = self.optimizer.domain_scaled
        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, bounds[:, 0], bounds[:, 1])

        all_f_min = []
        all_x_min = []

        def f_min_scipy_wrapper(x: np.ndarray):
            x = x.reshape(1, dim)
            means, vars = self.mfck.predict_all_levels(x)
            return means[level].item()

        for i in range(acq_x0.shape[0]):

            res = so.minimize(f_min_scipy_wrapper,
                              acq_x0[i, :],
                              method="L-BFGS-B",
                              bounds=bounds,
                              tol=4.4e-8)

            # TODO: check bounds -> apply L1 correction if applicable

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

    def epi(self, x: np.ndarray, level: int) -> np.ndarray:
        """
        Extended Probability of Improvement

        :param x:
        :param level:
        :return:
        """

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

        # probability of feasibility
        pof = 1.0

        for c_id in range(self.num_cstr):
            # g_pred = self.optimizer.cstr_surrogates[c_id].predict_values(x)
            # s2_pred = self.optimizer.cstr_surrogates[c_id].predict_variances(x)

            # TODO: add predict_all_levels() to mfck wrapper
            g_pred, s2_pred = self.optimizer.cstr_surrogates[c_id].mfck.predict_all_levels(x)

            pof *= stats.norm.cdf(-g_pred[level] / np.sqrt(s2_pred[level].reshape(1, 1)))

        return pi * corr * cost_ratio * density * pof

    def execute_infill_strategy(self, optimizer) -> list:

        self.acq_data = {}

        self.current_level = 0
        self.f_min, _ = self.predicted_f_min(-1)

        self.acq_data["model_fmin"] = self.f_min
        # self.acq_data["model_fmin_descaled"] = self.f_min * optimizer.yt[-1].std(axis=0) + optimizer.yt[-1].mean()

        self.num_dim = optimizer.num_dim
        self.bounds = optimizer.domain_scaled

        self.num_cstr = optimizer.num_cstr

        self.num_levels = optimizer.num_levels
        self.costs = optimizer.costs

        self.optimizer = optimizer

        x_infill, fid_infill = self.minimize_with_scipy()

        infill = []
        for lvl in range(self.num_levels):
            if lvl == fid_infill:
                infill.append(x_infill)
            else:
                infill.append(None)

        self.acq_data["infill"] = infill

        self.optimizer.iter_data["acquisition"] = self.acq_data

        return infill


    def minimize_with_scipy(self) -> tuple:

        self.current_level = 0

        def epi_scipy_wrapper(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            epi = self.epi(x, self.current_level).item()
            return -epi

        dim = self.optimizer.num_dim
        bounds = self.optimizer.domain_scaled

        acq_multistart = self.n_start
        acq_sampler = stats.qmc.LatinHypercube(d=self.num_dim)   # To be verified, but I believe scipy LHS sampler works better

        acq_x0 = acq_sampler.random(acq_multistart)
        acq_x0 = stats.qmc.scale(acq_x0, self.bounds[:, 0], self.bounds[:, 1])

        all_epi_f = []
        all_epi_x = []
        all_epi_level = []
        all_epi_success = 0
        all_epi_nit = []

        for lvl in range(self.num_levels):
            self.current_level = lvl

            for i in range(acq_x0.shape[0]):

                res = so.minimize(epi_scipy_wrapper,
                                  acq_x0[i, :],
                                  method="L-BFGS-B",
                                  bounds=bounds,
                                  jac="3-point",
                                  tol=4.4e-8)

                # check bounds -> apply L1 correction if necessary

                all_epi_f.append(-res.fun)
                all_epi_x.append(res.x)
                all_epi_level.append(self.current_level)

                if res.success:
                    all_epi_success += 1

                all_epi_nit.append(res.nit)

        all_epi_success /= (2*acq_x0.shape[0])

        idx = np.argmax(all_epi_f)
        x_infill = all_epi_x[idx]
        level_infill = all_epi_level[idx]

        self.acq_data["epi_f"] = all_epi_f
        self.acq_data["epi_x"] = all_epi_x
        self.acq_data["epi_level"] = all_epi_level
        self.acq_data["epi_success"] = all_epi_success
        self.acq_data["epi_nit"] = all_epi_nit

        return x_infill, level_infill


class VFEI(AcquisitionStrategy):

    def __init__(self, optimizer=None, **kwargs):
        super().__init__()

        self.optimizer = optimizer
        # self.acq_func = acq_func
        self.n_start = kwargs.get("n_start", None)

        self.f_min = np.inf

        if self.optimizer is not None:
            self.mfck = self.optimizer.obj_surrogate.mfck
        else:
            self.mfck = None

        if self.optimizer is not None and self.n_start is None:
            self.n_start = 10*self.optimizer.num_dim

    def compatibility_check(self, optimizer):
        raise Exception("Compatibility check not implemented.")

    def execute_infill_strategy(self, optimizer) -> list:

        self.optimizer = optimizer
        self.num_levels = self.optimizer.num_levels

        self.acq_data = {}

        self.fmin = self.get_fmin(self.optimizer)

        self.acq_data["fmin"] = self.fmin
        self.acq_data["fmin_descaled"] = self.fmin * optimizer.yt[-1].std(axis=0) + optimizer.yt[-1].mean()

        # sample the domain using a higher number of points for better starting locations
        sampler = stats.qmc.LatinHypercube(d=self.optimizer.num_dim)
        acq_x0 = sampler.random(10*self.n_start)
        acq_x0 = stats.qmc.scale(acq_x0, optimizer.domain_scaled[:, 0], optimizer.domain_scaled[:, 1])

        acq_f = self.ei_vf(acq_x0, 0) * self.probability_of_feasibility(acq_x0, 0)
        self.acq_data["max_init_acq_f"] = acq_f.max()
        sorted_indices = np.argsort(acq_f.ravel())[-self.n_start:]

        acq_x0 = acq_x0[sorted_indices, :]
        acq_x0 = acq_x0[-self.n_start:, :]

        acq_f = []
        acq_x = []
        acq_lvl = []

        for lvl in range(self.num_levels):

            def scipy_ei_vf_wrapper(x: np.ndarray) -> float:
                x = x.reshape(1, -1)
                ei_vf = self.ei_vf(x, lvl)
                pof = self.probability_of_feasibility(x, lvl)
                return -(ei_vf*pof).item()

            for i in range(acq_x0.shape[0]):

                res = so.minimize(scipy_ei_vf_wrapper,
                                  x0=acq_x0[i, :],
                                  method="L-BFGS-B",
                                  bounds=self.optimizer.domain_scaled,
                                  tol=1e-8,)

                # TODO: check bounds

                acq_f.append(-res.fun)
                acq_x.append(res.x)
                acq_lvl.append(lvl)

        idx = np.argmax(acq_f)
        x_infill = acq_x[idx]
        level_infill = acq_lvl[idx]

        self.acq_data["acq_f"] = acq_f
        self.acq_data["acq_x"] = acq_x
        self.acq_data["acq_lvl"] = acq_lvl

        infill = []

        for lvl in range(self.num_levels):
            if lvl == level_infill:
                infill.append(x_infill)
            else:
                infill.append(None)

        optimizer.iter_data["acquisition"] = self.acq_data

        return infill


    def ei_vf(self, x: np.ndarray, level: int):

        mfck = self.optimizer.obj_surrogate.mfck

        mu, s2 = mfck.predict_all_levels(x)
        rho_values = mfck.optimal_theta[2 + 2 * mfck.nx:: mfck.nx + 2]
        eta = mfck.eta(level, self.num_levels, rho_values)

        mu = mu[-1].reshape(-1, 1)                  # always use the highest level mean prediction
        s2 = eta**2 * s2[level].reshape(-1, 1)      # use the scaled level variance prediction

        ei = expected_improvement(mu, s2, self.fmin)

        return ei


    def probability_of_feasibility(self, x: np.ndarray, level: int):

        pof = np.ones((x.shape[0], 1))

        # do not compute PoF if the problem is unconstrained
        if self.optimizer.num_cstr == 0:
            return pof

        for c_id, c_surrogate in enumerate(self.optimizer.cstr_surrogates):
            mfck = c_surrogate.mfck

            mu, s2 = mfck.predict_all_levels(x)
            rho_values = mfck.optimal_theta[2 + 2 * mfck.nx:: mfck.nx + 2]
            eta = mfck.eta(level, self.num_levels, rho_values)

            mu = mu[-1].reshape(-1, 1)
            s2 = eta**2 * s2[level].reshape(-1, 1)

            pof *= stats.norm.cdf(-mu/np.sqrt(s2))

        return pof


    # TODO: generalize method (make static)
    def compute_rscv(self, cstr_array: np.ndarray, cstr_config: list, g_tol: float = 0., h_tol: float = 0.) -> np.ndarray:

        scv = np.full_like(cstr_array, 0.0)     # Square Constraint Violation

        for c_id, c_config in enumerate(cstr_config):

            if c_config.type in ["less", "greater"]:
                valid_mask = cstr_array[:, c_id] <= g_tol
                scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

            elif c_config.type == "equal":
                valid_mask = np.abs(cstr_array[:, c_id]) <= h_tol
                scv[~valid_mask, c_id] = cstr_array[~valid_mask, c_id]**2

            else:
                raise Exception(f"{c_config.type} is not a valid constraint type. It must be 'less', 'greater' or 'equal'.")

        rscv = np.sqrt(scv.sum(axis=1))

        return rscv

    # TODO: generelize method (make static)
    def get_fmin(self, optimizer, rscv_tol: float = 0.0) -> float:

        # no constraint
        if optimizer.num_cstr == 0:
            idx = optimizer.yt_scaled[-1].argmin()

        # with constraints
        if optimizer.num_cstr >= 1:
            ct_rscv = self.compute_rscv(optimizer.ct_scaled[-1], optimizer.cstr_config)
            feas_mask = np.where(ct_rscv <= rscv_tol, True, False)

            if np.any(feas_mask):
                idx = np.argmin(np.where(feas_mask, optimizer.yt_scaled[-1][:, 0], np.inf))
            else:
                idx = np.argmin(ct_rscv)

        fmin = optimizer.yt_scaled[-1][idx, 0]

        return fmin
