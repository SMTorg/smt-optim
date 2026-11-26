import numpy as np

from smt_optim.core.state import State
from smt_optim.acquisition_strategies.sego import SEGO


class qSEGO(SEGO):
    """
    qSEGO batch acquisition strategy.

    At each iteration, a batch of `batch_size` infill points is requested. The batch is found by
    `get_location`, which must return an array of shape (batch_size, num_dim). The fidelity
    selection and the assembly of the infill points are inherited from `SEGO`.

    Only mono-objective and continuous problems are supported. Constraints and multi-fidelity
    are handled by the `get_location` implementation (constraints) and by the inherited fidelity
    selection (multi-fidelity).

    .. note::
        `get_location` is not implemented yet and raises a `NotImplementedError`.

    Parameters
    ----------
    state : State
        Optimization state containing surrogate models, data, and problem definition.
    batch_size : int, optional
        Number of infill points per iteration. Default is 3.
    n_samples : int, optional
        Number of Monte Carlo scenarios used to estimate the batch acquisition function.
        Default is 512.
    n_start : int, optional
        Number of multistart initializations for the inner optimizer. Default is 20.
    sp_method : str, optional
        Optimization method passed to SciPy. Default is "L-BFGS-B".
    sp_tol : float, optional
        Tolerance for the SciPy optimizer. Default is sqrt(machine epsilon).
    seed : int or None, optional
        Random seed of the samplers. Default is None, which gives different draws on each run.

    Raises
    ------
    ValueError
        If the problem or the parameters are not supported (see `validate_config`).
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
        super().__init__(
            state,
            n_start=n_start,
            sp_method=sp_method,
            sp_tol=sp_tol,
            seed=seed,
        )

        self.batch_size = batch_size
        self.n_samples = n_samples

        self.validate_config(state)

    def validate_config(self, state: State) -> None:
        # TODO
        pass

    def get_location(self, state: State) -> np.ndarray:
        """
        Find the batch of locations to sample.

        Parameters
        ----------
        state : State
            Current optimization state, including surrogate models and data.

        Returns
        -------
        np.ndarray of shape (batch_size, num_dim)
            Batch of locations in the scaled design space.

        Raises
        ------
        NotImplementedError
            The batch acquisition optimization is not implemented yet.
        """

        raise NotImplementedError("qSEGO.get_location is not implemented yet.")
