import numpy as np
from scipy import optimize as so, stats as stats

from smt_optim.acquisition_functions import probability_of_improvement, fidelity_correlation
from smt_optim.acquisition_strategies import AcquisitionStrategy

from smt_optim.core.state import State

from smt_optim.utils.get_fmin import get_fmin

from smt_optim.subsolvers import multistart_minimize


class VFPI(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()
        self.n_start = kwargs.pop("n_start", None)
        self.apply_density_penalty = kwargs.pop("density_penalty", True)
        # self.cr_override = kwargs.pop("cr_override", None)                  # override optimizer Cost Ratio

        self.seed = kwargs.pop("seed", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if state and self.n_start is None:
            self.n_start = 10 * state.problem.num_dim

    def validate_config(self, acq_context: State) -> None:
        pass


    def get_infill(self, state: State) -> list[np.ndarray]:

        acq_data = dict()

        # get predicted f_min
        self.f_min = self.get_predicted_fmin(state)


        # get infill_x and infill_fidelity
        best_epi_f = -np.inf
        best_epi_x = None
        best_epi_lvl = None

        for lvl in range(state.problem.num_fidelity):
            # setup EPI
            func = lambda x, l=lvl, s=state: -self.epi(x, l, s)
            res = multistart_minimize(func,
                                      np.array([[0, 1]] * state.problem.num_dim),
                                      seed=self.seed)

            if -res.fun > best_epi_f:
                best_epi_f = -res.fun
                best_epi_x = res.x
                best_epi_lvl = lvl

        infills = []
        for lvl in range(state.problem.num_fidelity):
            if lvl == best_epi_lvl:
                infills.append(best_epi_x.copy().reshape(1, -1))
            else:
                infills.append(None)

        return infills



    def get_predicted_fmin(self, state):
        """
        Estimate the minimum predicted objective value using multistart optimization.

        This method constructs a wrapper around the first surrogate objective model
        stored in ``state.obj_models`` and performs a multistart optimization over
        the unit hypercube :math:`[0, 1]^d`, where :math:`d` is the problem dimension.
        The optimization is carried out using the ``multistart_minimize`` routine.

        Parameters
        ----------
        state : State
            Optimization state

        Returns
        -------
        float
            The minimum predicted objective function value found across all
            multistart runs.

        Notes
        -----
        - The search domain is assumed to be the unit hypercube.
        - The optimization relies on ``self.n_start`` initial points and
          ``self.seed`` for reproducibility.

        See Also
        --------
        multistart_minimize : Multistart optimization routine used to perform the search.
        """

        def obj_wrapper(x):
            y = state.obj_models[0].predict_values(x.reshape(1, -1))
            return y.item()

        res = multistart_minimize(obj_wrapper,
                                  np.array([[0, 1]] * state.problem.num_dim),
                                  n_start=self.n_start,
                                  seed=self.seed)

        return res.fun

    def epi(self, x: np.ndarray, lvl: int, state: State) -> float:
        """
        Evaluate the Extended Probability of Improvement (EPI) acquisition function.

        This method computes an acquisition value that combines the probability of
        improvement at the highest fidelity with several multiplicative correction
        factors accounting for fidelity correlation, evaluation cost, sampling
        density, and probability of feasibility (for constrained optimization).

        Parameters
        ----------
        x : ndarray of shape (1, n_dim)
            Input point at which the acquisition function is evaluated.
        lvl : int
            Fidelity level at which the acquisition function is computed.
            Lower values correspond to lower-fidelity (cheaper) models, starting from 0.
        state : State
            Optimization state

        Returns
        -------
        float
            Value of the EPI acquisition function at the given point and fidelity level.

        Notes
        -----
        The EPI criterion is defined as a product of the following terms:

        - **Probability of Improvement (PI)**:
          Computed at the highest fidelity level using the predictive mean and variance.
        - **Fidelity Correlation Penalty**:
          Accounts for the correlation between the selected fidelity level and the
          highest fidelity.
        - **Cost Ratio**:
          Ratio of highest-fidelity cost to the cost at the selected level.
        - **Density Penalty**:
          Optional factor that penalizes regions with high sampling density.
        - **Probability of Feasibility (PoF)**:
          Product of feasibility probabilities across all constraints, evaluated
          at the selected fidelity level.

        The input ``x`` is reshaped internally to match the expected input format
        of the surrogate models.

        See Also
        --------
        probability_of_improvement : Computes the probability of improvement.
        fidelity_correlation : Computes correlation between fidelity levels.
        sample_density : Estimates local sampling density (if enabled).
        """
        x = x.reshape(1, -1)

        mu, s2 = state.obj_models[0].model.predict_all_levels(x)
        cov = state.obj_models[0].predict_level_covariances(x, lvl)

        # probability of improvement
        pi = probability_of_improvement(mu[-1].reshape(-1, 1), s2[-1].reshape(-1, 1), self.f_min)

        # fidelity correlation penalty
        corr = fidelity_correlation(cov, s2[lvl].reshape(-1, 1), s2[-1].reshape(-1, 1))

        # cost ratio penalty
        cost_ratio = state.problem.costs[-1]/state.problem.costs[lvl]

        # density penalty
        density = 1.0
        if self.apply_density_penalty:
            density = self.sample_density(x, lvl, state.obj_models[0].model)

        # probability of feasibility
        pof = 1.0

        for c_id in range(state.problem.num_cstr):
            # g_pred = self.optimizer.cstr_surrogates[c_id].predict_values(x)
            # s2_pred = self.optimizer.cstr_surrogates[c_id].predict_variances(x)

            # TODO: add predict_all_levels() to mfck wrapper
            g_pred, s2_pred = state.cstr_models[c_id].model.predict_all_levels(x)

            pof *= stats.norm.cdf(-g_pred[lvl] / np.sqrt(s2_pred[lvl].reshape(1, 1)))

        return (pi * corr * cost_ratio * density * pof).item()


    def sample_density(self, x: np.ndarray, lvl: int, mfck) -> np.ndarray:
        """
        Compute a sampling density penalty based on correlation structure.

        This method evaluates a multiplicative penalty that reflects the local
        sampling density at a given point ``x`` for a specified fidelity level.
        The penalty is derived from the correlation kernel of a multi-fidelity
        co-Kriging (MFCK) model and decreases in regions where training samples
        are dense.

        Parameters
        ----------
        x : ndarray of shape (n_dim,) or (n_eval, n_dim)
            Input point(s) at which the density penalty is evaluated.
        lvl : int
            Fidelity level at which the density is computed.
        mfck : MFCK
            Multi-fidelity co-Kriging model from the SMT package.

        Returns
        -------
        ndarray of shape (n_eval, 1)
            Density penalty values at the input locations. Lower values indicate
            regions with higher sampling density.

        Notes
        -----
        - Inputs are internally normalized using the model's scaling and offset.
        - The kernel hyperparameters (``sigma2`` and ``theta``) are extracted
          from ``optimal_theta`` based on the fidelity level.
        - The penalty is computed as the product of
          :math:`1 - k(x, x_i) / \\sigma^2` over all training points
          :math:`x_i` at the specified level.
        - This formulation encourages exploration by penalizing regions that are
          strongly correlated with existing samples.
        """
        x_scale = mfck.X_scale
        x_offset = mfck.X_offset

        x = (x - x_offset) / x_scale

        xt_lvl = mfck.X[lvl]
        xt_lvl = (xt_lvl - x_offset) / x_scale
        dim = xt_lvl.shape[1]

        optimal_theta = mfck.optimal_theta

        if lvl == 0:
            sigma2 = optimal_theta[0]
            theta = optimal_theta[1:dim + 1]
        else:
            start = (dim + 1) + (2 + dim) * (lvl - 1)
            end = (dim + 1) + (2 + dim) * (lvl)
            sigma2 = optimal_theta[start]
            theta = optimal_theta[start + 1:end - 1]

        R = 1 - mfck._compute_K(x, xt_lvl, (sigma2, theta)) / sigma2
        penalty = np.prod(R, axis=1).reshape(-1, 1)

        return penalty

