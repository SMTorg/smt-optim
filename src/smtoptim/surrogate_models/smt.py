import copy
import warnings

import numpy as np

from smt.surrogate_models import KRG
from smt.applications import MFK, MFCK

from smt.design_space import (
    DesignSpace,
    CategoricalVariable,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)

from smtoptim.surrogate_models import Surrogate


EPSILON = np.finfo(float).eps


def check_theta_bounds(theta: np.ndarray, theta_bounds: np.ndarray) -> np.ndarray:
    """
    Apply corrections to GP hyperparameters that are on or outside of their boundaries,
    with a small overhead to avoid triggering SMT warnings.

    :param theta: GP hyperparameters.
    :type theta: np.ndarray

    :param theta_bounds: Hyperparameter bounds. np.ndarray[lower, upper].
    :type theta_bounds: np.ndarray

    :return: The corrected GP hyperparameters.
    :rtype: np.ndarray
    """
    lower_disc = (theta <= theta_bounds[0])
    theta = np.where(lower_disc, theta_bounds[0] + np.sqrt(EPSILON), theta)

    upper_disc = (theta >= theta_bounds[1])
    theta = np.where(upper_disc, theta_bounds[1] - np.sqrt(EPSILON), theta)

    return theta


def filter_nan_values(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray]:
    # TODO: filter inf and similar non numeric values and add doc
    valid_mask = ~np.isnan(y).ravel()
    x_val = x[valid_mask]
    y_val = y[valid_mask]
    return x_val, y_val

def clean_training_data(x: list[np.ndarray], y: list[np.ndarray]) -> tuple[np.ndarray]:
    # TODO: add doc
    num_level = len(x)

    for lvl in range(num_level):
        x[lvl], y[lvl] = filter_nan_values(x[lvl], y[lvl])

    return x, y


class SmtKRG(Surrogate):

    def __init__(self, optimizer=None, name=None):
        super().__init__()

        # TODO: implement a way to control the random_state

        self.optimizer = optimizer
        self.name = name

        self.num_dim = 0
        self.krg = None
        self.krg_initialized = False

        if optimizer is not None:
            self._init_smt(optimizer)

        # TODO: should use optimizer.domain

        else:
            pass
            # raise Exception("Unsupported domain type.")

    def _init_smt(self, optimizer):

        self.num_dim = optimizer.num_dim
        self.costs = optimizer.costs

        self.n_start = 3*optimizer.num_dim
        self.previous_theta = np.ones(self.num_dim)

        # KRG for continuous domain
        if type(self.optimizer.design_space) is np.ndarray:
            self.krg = KRG(print_global=False,
                           n_start=self.n_start,
                           hyper_opt="Cobyla",
                           random_state=None)

        # KRG for mixed integer domain
        # elif type(optimizer.domain) is DesignSpace:
        #     self.dim = optimizer.domain.n_dv
        #     self.krg = KRG(design_space=domain,
        #                     categorical_kernel=MixIntKernelType.CONT_RELAX,
        #                     hyper_opt="Cobyla",
        #                     corr="abs_exp",
        #                     n_start=3*self.dim,
        #                     print_global=False,
        #     )

            self.theta_bounds = self.krg.options["theta_bounds"]
        else:
            pass
            # raise Exception("Unsupported domain type.")

        self.krg_initialized = True

    def train(self, xt: list, yt: list, **kwargs):
        """
        Train the GP on the training data.

        Args:
            xt (list[np.ndarray]): training data variables
            yt (list[np.ndarray]): training data values
        """

        # if not self.krg_initialized:
        #     raise Exception("KRG must be initialized before training.")

        # print(f"xt= \n{xt}")
        try:
            if type(self.optimizer.design_space) is np.ndarray:
                self.previous_theta = check_theta_bounds(self.previous_theta, self.theta_bounds)
                self.krg.options["theta0"] = self.previous_theta
                self.krg.options["n_start"] = self.n_start
        except:
            warnings.warn("Error changing KRG parameters.")


        self.xt = xt[-1].copy()
        self.yt = yt[-1].copy()
        self.xt, self.yt = filter_nan_values(self.xt, self.yt)

        self.krg.set_training_values(self.xt, self.yt)
        self.krg.train()

        # store the optimize theta vector for the next iteration
        self.previous_theta = self.krg.optimal_theta
        if self.name: self.optimizer.iter_data[f"{self.name}_opt_theta"] = self.previous_theta

    def predict_values(self, x_pred):
        y_pred = self.krg.predict_values(x_pred)
        return y_pred

    def predict_variances(self, x_pred):
        s2_pred = self.krg.predict_variances(x_pred)
        return s2_pred

class SmtAutoModel(Surrogate):

    def __init__(self):
        super().__init__()
        self.model = None
        pass

    def train(self, xt: list[np.ndarray], yt: list[np.ndarray], **kwargs) -> None:
        """
        Train the GP on the training data.

        Args:
            xt (list[np.ndarray]): training data variables
            yt (list[np.ndarray]): training data values
        """

        num_dim = xt[-1].shape[1]
        num_fidelity = len(xt)

        if num_fidelity == 1:
            self.model = KRG(print_global=False, n_start=3*num_dim, hyper_opt="Cobyla", seed=42)
        else:
            self.model = MFK(print_global=False, n_start=3*num_dim, hyper_opt="Cobyla", seed=42)

            for lvl in range(num_fidelity-1):
                self.model.set_training_values(xt[lvl], yt[lvl], name=lvl)

        self.model.set_training_values(xt[-1], yt[-1])
        self.model.train()

    def predict_values(self, x_pred: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict_values(x_pred)
        return y_pred

    def predict_variances(self, x_pred: np.ndarray) -> np.ndarray:
        s2_pred = self.model.predict_variances(x_pred)
        return s2_pred


class SmtMFK(Surrogate):

    def __init__(self, optimizer=None, name=None):

        self.optimizer = optimizer

        self.name = name

        self.num_dim = 0
        self.num_levels = 0
        self.costs = []
        self.mfk = None
        self.mfk_initialized = False

        if optimizer is not None:
            self._init_smt(optimizer)


    def _init_smt(self, optimizer):

        self.num_dim = optimizer.num_dim
        self.num_levels = optimizer.num_fidelity
        self.costs = optimizer.costs

        self.n_start = 3*optimizer.num_dim
        self.previous_theta = np.ones((self.num_levels, self.num_dim))

        self.mfk = MFK(print_global=False,
                       n_start=self.n_start,
                       hyper_opt="Cobyla",
                       random_state=None)

        self.theta_bounds = self.mfk.options["theta_bounds"]

        self.mfk_initialized = True


    def train(self, xt: list[np.ndarray], yt: list[np.ndarray], **kwargs) -> None:

        if not self.mfk_initialized:
            raise Exception("MFK must be initialized before training.")

        try:
            self.previous_theta = check_theta_bounds(self.previous_theta, self.theta_bounds)
            self.mfk.options["theta0"] = self.previous_theta
            self.mfk.options["n_start"] = self.n_start
        except:
            warn("Error changing MFK parameters.")

        # TODO: data should be cleaned in the Driver class
        self.xt = copy.deepcopy(xt)
        self.yt = copy.deepcopy(yt)
        self.xt, self.yt = clean_training_data(self.xt, self.yt)

        for k in range(self.num_levels-1):
            self.mfk.set_training_values(self.xt[k], self.yt[k], name=k)

        self.mfk.set_training_values(self.xt[-1], self.yt[-1])

        self.mfk.train()

        self.previous_theta = np.array(self.mfk.optimal_theta).reshape(self.num_levels, self.num_dim)

        if self.name: self.optimizer.iter_data[f"{self.name}_opt_theta"] = self.previous_theta

    def predict_values(self, x_pred: np.ndarray) -> np.ndarray:
        y_pred = self.mfk.predict_values(x_pred)
        return y_pred

    def predict_variances(self, x_pred: np.ndarray) -> np.ndarray:
        s2_pred = self.mfk.predict_variances(x_pred)
        return s2_pred

    def predict_variances_all_levels(self, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the uncertainty reduction and the square scale factor of each level.

        Args:
            x_pred (np.ndarray): coordinates of the next infill location

        Returns:
            s2_red (np.ndarray): uncertainty reduction of each fidelity level
            rho2 (np.ndarray): square scale factor between each fidelity level

        """

        # np.ndarray(num_points, num_levels), list[np.ndarray(num_points)]
        s2_pred, rho2 = self.mfk.predict_variances_all_levels(x_pred)
        s2_red = np.zeros(self.num_levels)

        # s2_red[0] = s2_pred[0, 0]

        # for k in range(1, self.num_levels):
        #     s2_red[k] = s2_pred[0, k] - rho2[k-1].item() * s2_pred[0, k-1]

        tot_rho2 = np.ones(self.num_levels-1)

        # TODO: add adjust variance reduction computation to account for the nugget

        for k in range(self.num_levels-1):
            tot_rho2[k] = 1
            for l in range(k, self.num_levels-1):
                tot_rho2[k] *= rho2[l].item()

            s2_red[k] = s2_pred[0, k]*tot_rho2[k]

        s2_red[-1] = s2_pred[0, -1]

        return s2_red, tot_rho2

class SmtMFCK(Surrogate):

    def __init__(self, optimizer=None, name=None):

        self.optimizer = optimizer

        self.name = name

        self.num_dim = 0
        self.num_levels = 0
        self.costs = []
        self.mfck = None
        self.mfck_initialized = False

        if optimizer is not None:
            self._init_smt(optimizer)

    def _init_smt(self, optimizer):

        self.num_dim = optimizer.num_dim
        self.num_levels = optimizer.num_fidelity
        self.costs = optimizer.costs

        self.n_start = 3*optimizer.num_dim

        self.mfck = MFCK(print_global=False,
                         n_start=self.n_start,
                         hyper_opt="Cobyla",
                         random_state=None)

        self.mfck.options["lambda"] = 0.0

        self.mfck_initialized = True

    def train(self, xt: list, yt: list, **kwargs):

        if not self.mfck_initialized:
            raise Exception("MFK must be initialized before training.")

        self.xt = xt
        self.yt = yt

        for k in range(self.num_levels-1):
            self.mfck.set_training_values(self.xt[k], self.yt[k], name=k)

        self.mfck.set_training_values(self.xt[-1], self.yt[-1])

        self.mfck.train()

        # self.previous_theta = np.array(self.mfk.optimal_theta).reshape(self.num_levels, self.dim)

        # if self.name: self.optimizer.iter_data[f"{self.name}_opt_theta"] = self.previous_theta

    def predict_values(self, x_pred: np.ndarray) -> np.ndarray:
        y_pred = self.mfck.predict_values(x_pred)
        return y_pred

    def predict_variances(self, x_pred: np.ndarray) -> np.ndarray:
        s2_pred = self.mfck.predict_variances(x_pred)
        return s2_pred
