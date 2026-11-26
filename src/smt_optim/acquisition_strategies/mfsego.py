"""
Backward-compatibility module: `MFSEGO` has been renamed `SEGO` and moved to
`smt_optim.acquisition_strategies.sego`.

Everything defined in `sego.py` is re-exported here so that old imports keep working.
"""

from .sego import *  # noqa: F401,F403
from .sego import SEGO

MFSEGO = SEGO
