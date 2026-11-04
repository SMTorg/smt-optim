import numpy as np
from scipy import stats
import smt.design_space as ds

from smt_optim.acquisition_functions import q_log_ei
from smt_optim.acquisition_strategies import AcquisitionStrategy
from smt_optim.core.state import State
from smt_optim.subsolvers import multistart_minimize


class qLogEI(AcquisitionStrategy):
    """
    qLogEI batch acquisition strategy.

    At each iteration, all the points of the batch are optimized jointly with a multistart
    optimization of qLogEI.

    Only unconstrained, mono-objective, mono-fidelity and continuous problems are supported.

    Parameters
    ----------
    state : State
        Optimization state containing surrogate models, data, and problem definition.

    Other Parameters
    ----------------
    batch_size: int, optional
        Number of infill points per iteration. Default is 3.
    n_samples: int, optional
        Number of scenarios of qLogEI. Default is 512.
    n_start: int, optional
        Number of multistart initializations for the inner optimizer. Default is 20.
    sp_method: str, optional
        Optimization method passed to SciPy. Default is "L-BFGS-B".
    sp_tol: float, optional
        Tolerance for the SciPy optimizer. Default is sqrt(machine epsilon).
    seed: int, optional
        Random seed of the samplers. Default is None, which gives different draws on each
        run.
    """

    def __init__(
        self,
        state: State,
        batch_size: int = 3,
        n_samples: int = 512,
        n_start: int = 20,
        sp_method: str = "L-BFGS-B",
        sp_tol: float = np.sqrt(np.finfo(float).eps),
        seed: int | None = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_start = n_start
        self.sp_method = sp_method
        self.sp_tol = sp_tol

        rng = np.random.default_rng(seed)
        self.z_sampler = stats.qmc.LatinHypercube(d=batch_size, seed=rng)
        self.x0_sampler = stats.qmc.LatinHypercube(
            d=batch_size * state.problem.num_dim, seed=rng
        )

    def validate_config(self, state: State) -> None:

        if state.problem.num_obj > 1:
            raise Exception("Multi-objective not implemented.")

        if state.problem.num_cstr > 0:
            raise Exception("Constraints not implemented.")

        if state.problem.num_fidelity > 1:
            raise Exception("Multi-fidelity not implemented.")

        for dv in state.problem.design_space.design_variables:
            if not isinstance(dv, ds.FloatVariable):
                raise Exception("Mixed variables not implemented.")

        obj_required_methods = [
            "predict_values",
            "predict_covariances",
        ]

        for method in obj_required_methods:
            if not callable(getattr(state.obj_models[0], method, None)):
                raise TypeError(f"Objective model requires: '{method}' method.")

    def get_infill(self, state: State) -> list[np.ndarray]:
        """
        Compute the next batch of infill points.

        Parameters
        ----------
        state : State
            Current optimization state, including surrogate models and data.

        Returns
        -------
        list of ndarray
            Single-element list containing the batch of shape (batch_size, num_dim).
        """

        self.validate_config(state)

        acq_data = dict()

        model = state.obj_models[0]
        num_var = self.batch_size * state.problem.num_dim
        f_min = state.get_best_sample(ctol=0.0, scaled=True).obj[0]

        # scenarios kept fixed during the optimization
        z = stats.norm.ppf(self.z_sampler.random(self.n_samples))

        def scipy_acq_func(x):
            batch = x.reshape(self.batch_size, -1)
            mu = model.predict_values(batch)
            cov = model.predict_covariances(batch)
            return -q_log_ei(mu, cov, f_min, z)

        # generate starting batches for the multistart optimization
        multi_x0 = self.x0_sampler.random(self.n_start)

        res = multistart_minimize(
            scipy_acq_func,
            bounds=np.array([[0, 1]] * num_var),
            multi_x0=multi_x0,
            tol=self.sp_tol,
            method=self.sp_method,
        )

        acq_data["qlogei"] = -res.fun

        state.iter_log["acquisition"] = acq_data

        return [res.x.reshape(self.batch_size, -1)]
