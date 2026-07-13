import numpy as np
from smt_optim.core import Sample
from smt_optim.utils.constraints import compute_rscv

def InjectData(state, xt, yt, ct=None):
    """
    Injects matrix-form data (X, Y, and optionally Constraints) into the State.
    """
    # 1. Ensure inputs are 2D arrays
    xt = np.atleast_2d(xt)
    yt = np.atleast_2d(yt)
    
    # Handle constraints
    num_samples = xt.shape[0]
    if ct is None:
        ct = np.empty((num_samples, 0))
    else:
        ct = np.atleast_2d(ct)

    # 2. Add to dataset
    for i in range(num_samples):
        # We compute RSCV because export_as_dict will crash without it
        rscv_val = compute_rscv(ct[i:i+1, :], state.problem.cstr_configs).item()
        
        sample = Sample(
            x=xt[i],
            fidelity=0,
            obj=yt[i],
            cstr=ct[i] if ct.size > 0 else np.array([]),
            eval_time=np.zeros(yt.shape[1] + ct.shape[1]),
            metadata={"rscv": rscv_val, "iter": 0, "budget": 0.0}
        )
        state.dataset.add(sample)

    # Update state metadata if necessary
    state.iter = 0
