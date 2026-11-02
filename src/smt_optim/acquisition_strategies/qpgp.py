import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


from smt_optim.acquisition_functions import integrated_variance_reduction, vec_mean_ivr
from smt_optim.acquisition_strategies import AcquisitionStrategy
from smt_optim.core.state import State
from smt_optim.utils.multi_obj import PymooStateWrapper


class qPGP(AcquisitionStrategy):
    """
    qPGP batch acquisition strategy.

    At each iteration, the bi-objective mean / Integrated Variance Reduction acquisition
    function min(mu, -IVR) is solved on the surrogate model with NSGA-II. The Pareto point with
    the lowest mean prediction is always included in the batch. The rest of the Pareto set is
    then clustered in the input space with k-means, and one candidate per cluster is selected
    to complete the batch of `batch_size` infill points.

    Only unconstrained, mono-objective and mono-fidelity problems are supported.

    Parameters
    ----------
    state : State
        Optimization state containing surrogate models, data, and problem definition.

    Other Parameters
    ----------------
    batch_size: int, optional
        Number of infill points per iteration. Default is 3.
    pop_size: int, optional
        NSGA-II population size. Default is 200.
    n_gen: int, optional
        NSGA-II number of generations. Default is 100.
    crossover_prob: float, optional
        SBX crossover probability. Default is 0.9.
    eta: float, optional
        Distribution index of the SBX crossover and polynomial mutation. Default is 20.
    n_integration: int, optional
        Number of Sobol integration points used by IVR. Default is 64.
    seed: int, optional
        Random seed for reproducibility. Default is None.

    Notes
    -----
    Based on the global phase of STREGO (Algorithm 2 in [1]_).

    References
    ----------
    .. [1] Diouane, Y., El Hallaoui, I., Kadri, M., Lamghari, A. (2026). STREGO: A Scalable
       Implementation of a Trust-Region Framework for Noisy Bayesian Optimization.
       arXiv:2610.14338. https://arxiv.org/abs/2610.14338
    """

    def __init__(
        self,
        state: State,
        batch_size: int = 3,
        pop_size: int = 200,
        n_gen: int = 100,
        crossover_prob: float = 0.9,
        eta: float = 20.0,
        n_integration: int = 64,
        seed: int | None = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.eta = eta
        self.n_integration = n_integration
        self.seed = seed

        self.validate_config(state)

    def validate_config(self, state: State) -> None:

        if state.problem.num_obj > 1:
            raise ValueError("qPGP only supports mono-objective problems.")

        if state.problem.num_cstr > 0:
            raise ValueError("qPGP only supports unconstrained problems.")

        if state.problem.num_fidelity > 1:
            raise ValueError("qPGP only supports mono-fidelity problems.")

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

        if self.seed is not None:
            self.seed += 1

        acq_data = dict()

        # integration points shared by all the IVR evaluations of the iteration
        sampler = stats.qmc.Sobol(d=state.problem.num_dim, seed=state.iter)
        integration_points = sampler.random(self.n_integration)

        # solve min(mu, -IVR) with NSGA-II
        pymoo_prob = MeanIVRProblem(state, integration_points)
        algorithm = NSGA2(
            pop_size=self.pop_size,
            crossover=SBX(prob=self.crossover_prob, eta=self.eta),
            mutation=PM(eta=self.eta),
            seed=self.seed,
        )
        res = minimize(pymoo_prob, algorithm, ("n_gen", self.n_gen), seed=self.seed)

        # Pareto set
        pareto_x = np.atleast_2d(res.X)
        pareto_mu = np.atleast_2d(res.F)[:, 0]

        acq_data["pareto_size"] = pareto_x.shape[0]

        # lowest mean point + clustering of the rest of the Pareto set in the input space
        batch = select_batch(pareto_x, pareto_mu, self.batch_size, seed=self.seed)

        state.iter_log["acquisition"] = acq_data

        return [batch]


class MeanIVRProblem(PymooStateWrapper):
    def __init__(self, state: State, integration_points: np.ndarray):
        """
        Pymoo problem of the bi-objective min(mu, -IVR) acquisition function.

        Parameters
        ----------
        state : State
            Optimization state with trained surrogate models.
        integration_points : np.ndarray
            Monte-Carlo integration points for IVR of shape (N_mc, num_dim).

        Notes
        -----
        Extends `PymooStateWrapper` by replacing the objectives with the mean / IVR acquisition
        function `vec_mean_ivr` of the objective surrogate model, evaluated on the whole
        population at once. Uses the scaled design space.

        The integration points are provided by the strategy so that all the IVR evaluations
        of an iteration share the same integration grid.
        """

        super().__init__(state, scaled=True, train=False)

        self.integration_points = integration_points
        self.model = state.obj_models[0]
        self.n_obj = 2

    def _evaluate(self, x, out, *args, **kwargs):

        x_scaled = (x - self.l_bounds) / (self.u_bounds - self.l_bounds)

        mu = self.model.predict_values(x_scaled)
        ivr = integrated_variance_reduction(
            self.model, x_scaled, self.integration_points
        )

        out["F"] = vec_mean_ivr(mu, ivr)


def select_batch(
    pareto_x: np.ndarray,
    pareto_mu: np.ndarray,
    batch_size: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Select a batch of points from the Pareto set: the point with the lowest mean prediction,
    then `batch_size - 1` points by k-means clustering of the rest of the Pareto set in the
    input space.

    Parameters
    ----------
    pareto_x : np.ndarray
        Pareto set of shape (num_points, num_dim).
    pareto_mu : np.ndarray
        Mean predictions of the Pareto set of shape (num_points,).
    batch_size : int
        Number of points to select.
    seed : int, optional
        Random seed of the k-means initialization.

    Returns
    -------
    np.ndarray
        Selected points of shape (min(batch_size, num_unique_points), num_dim). The first
        point is the one with the lowest mean prediction.

    Notes
    -----
    For each cluster, the Pareto point closest to the centroid is selected. If some clusters
    are empty, the batch is completed with the points farthest from the already selected ones.
    If `batch_size` is 1, only the lowest mean point is selected.
    """

    best = pareto_x[np.argmin(pareto_mu)]

    # remaining unique Pareto points (duplicates of the lowest mean point removed)
    others = np.unique(pareto_x, axis=0)
    others = others[~np.all(others == best, axis=1)]

    n_clusters = batch_size - 1

    if n_clusters == 0:
        return best[None, :]

    if others.shape[0] <= n_clusters:
        return np.vstack([best, others])

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(others, n_clusters, minit="++", seed=rng)

    selected = []
    for k in np.unique(labels):
        members = np.where(labels == k)[0]
        dist = np.linalg.norm(others[members] - centroids[k], axis=1)
        selected.append(members[np.argmin(dist)])

    # complete the batch if some clusters are empty (maximin distance, lowest mean point included)
    while len(selected) < n_clusters:
        chosen = np.vstack([best, others[selected]])
        min_dist = cdist(others, chosen).min(axis=1)
        selected.append(int(np.argmax(min_dist)))

    return np.vstack([best, others[selected]])
