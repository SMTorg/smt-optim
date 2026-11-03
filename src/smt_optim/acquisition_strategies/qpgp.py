from typing import Callable

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


from smt_optim.acquisition_functions import init_bi_acquisition
from smt_optim.acquisition_strategies import AcquisitionStrategy
from smt_optim.core.state import State
from smt_optim.utils.multi_obj import PymooStateWrapper


class qPGP(AcquisitionStrategy):
    """
    qPGP batch acquisition strategy.

    At each iteration, the bi-objective minimization problem min(acq_1, acq_2) is solved on the
    surrogate model with NSGA-II. The Pareto set is then clustered in the input space with
    k-means, and one candidate per cluster is selected to form the batch of `batch_size` infill
    points.

    Only unconstrained, mono-objective and mono-fidelity problems are supported.

    Parameters
    ----------
    state : State
        Optimization state containing surrogate models, data, and problem definition.
    acq_init_1 : Callable
        Initializer of the first acquisition function, to be minimized. See
        `init_bi_acquisition`.
    acq_init_2 : Callable
        Initializer of the second acquisition function, to be minimized. See
        `init_bi_acquisition`.

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
    seed: int, optional
        Random seed for reproducibility. Default is None.

    Notes
    -----
    The acquisition functions are initialized at each iteration, as the surrogate models are
    retrained. They take points of shape (num_points, num_dim) in the scaled design space.

    References
    ----------
    .. [1] Diouane, Y., El Hallaoui, I., Kadri, M., Lamghari, A. (2026). STREGO: A Scalable
       Implementation of a Trust-Region Framework for Noisy Bayesian Optimization.
       arXiv:2610.14338. https://arxiv.org/abs/2610.14338

    Examples
    --------
    Bi-objective minimization of the mean prediction and the negative IVR, as in the global
    phase of STREGO (Algorithm 2 in [1]_):

    >>> from smt_optim.acquisition_functions import init_ivr, init_mean, negate
    >>> driver = Driver(
    ...     problem=problem,
    ...     config=config,
    ...     strategy=qPGP,
    ...     strategy_kwargs=dict(
    ...         acq_init_1=init_mean,
    ...         acq_init_2=negate(init_ivr),
    ...         batch_size=4,
    ...     ),
    ... )
    >>> state = driver.optimize()
    """

    def __init__(
        self,
        state: State,
        acq_init_1: Callable,
        acq_init_2: Callable,
        batch_size: int = 3,
        pop_size: int = 200,
        n_gen: int = 100,
        crossover_prob: float = 0.9,
        eta: float = 20.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.acq_init_1 = acq_init_1
        self.acq_init_2 = acq_init_2
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.eta = eta
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
            Single-element list containing the batch of shape
            (min(batch_size, num_unique_pareto_points), num_dim).
        """

        if self.seed is not None:
            self.seed += 1

        acq_data = dict()

        # solve min(acq_1, acq_2) with NSGA-II
        bi_acquisition = init_bi_acquisition(state, self.acq_init_1, self.acq_init_2)
        pymoo_prob = BiAcquisitionProblem(state, bi_acquisition)
        algorithm = NSGA2(
            pop_size=self.pop_size,
            crossover=SBX(prob=self.crossover_prob, eta=self.eta),
            mutation=PM(eta=self.eta),
            seed=self.seed,
        )
        res = minimize(pymoo_prob, algorithm, ("n_gen", self.n_gen), seed=self.seed)

        # Pareto set
        pareto_x = np.atleast_2d(res.X)

        acq_data["pareto_size"] = pareto_x.shape[0]

        # clustering of the Pareto set in the input space
        batch = select_batch(pareto_x, self.batch_size, seed=self.seed)

        state.iter_log["acquisition"] = acq_data

        return [batch]


class BiAcquisitionProblem(PymooStateWrapper):
    def __init__(self, state: State, bi_acquisition: Callable):
        """
        Pymoo problem of the bi-objective minimization of two acquisition functions.

        Parameters
        ----------
        state : State
            Optimization state with trained surrogate models.
        bi_acquisition : Callable
            Function of the points x of shape (num_points, num_dim) returning the values of the
            two acquisition functions of shape (num_points, 2), both to be minimized.

        Notes
        -----
        Extends `PymooStateWrapper` by replacing the objectives with `bi_acquisition`, evaluated
        on the whole population at once. Uses the scaled design space.
        """

        super().__init__(state, scaled=True, train=False)

        self.bi_acquisition = bi_acquisition
        self.n_obj = 2

    def _evaluate(self, x, out, *args, **kwargs):

        x_scaled = (x - self.l_bounds) / (self.u_bounds - self.l_bounds)

        out["F"] = self.bi_acquisition(x_scaled)


def select_batch(
    pareto_x: np.ndarray,
    batch_size: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Select a batch of points from the Pareto set by k-means clustering in the input space.

    Parameters
    ----------
    pareto_x : np.ndarray
        Pareto set of shape (num_points, num_dim).
    batch_size : int
        Number of points to select.
    seed : int, optional
        Random seed of the k-means initialization.

    Returns
    -------
    np.ndarray
        Selected points of shape (min(batch_size, num_unique_points), num_dim).

    Notes
    -----
    For each cluster, the Pareto point closest to the centroid is selected. If some clusters
    are empty, the batch is completed with the points farthest from the already selected ones.
    """

    points = np.unique(pareto_x, axis=0)

    if points.shape[0] <= batch_size:
        return points

    centroids, labels = kmeans2(points, batch_size, minit="++", seed=seed)

    selected = []
    for k in np.unique(labels):
        members = np.where(labels == k)[0]
        dist = np.linalg.norm(points[members] - centroids[k], axis=1)
        selected.append(members[np.argmin(dist)])

    # complete the batch if some clusters are empty (maximin distance)
    while len(selected) < batch_size:
        min_dist = cdist(points, points[selected]).min(axis=1)
        selected.append(int(np.argmax(min_dist)))

    return points[selected]
