from typing import Callable
import numpy as np

from smt_optim.core import State

AcqFunc = Callable[[np.ndarray], np.ndarray]  # (n, d) -> (n, 1), higher is better
AcqInit = Callable[[State], AcqFunc]


def legacy_acq_init(acq_func):
    def init(state):
        fmin = state.get_best_sample(ctol=0.0, scaled=True).obj[0]
        model = state.obj_models[0]

        def acq(x):
            mu = model.predict_values(x)
            s2 = model.predict_variances(x)
            return np.array(
                [[acq_func(m, v, fmin)] for m, v in zip(mu.ravel(), s2.ravel())]
            )

        return acq

    return init
