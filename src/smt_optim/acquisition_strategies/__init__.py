from .base import AcquisitionStrategy
from .sego import SEGO
from .mfsego import MFSEGO  # backward compatibility alias of SEGO
from .vfpi import VFPI
from .mosego import MOSEGO
from .bisego import BiSEGO
from .qpgp import qPGP

__all__ = [
    "AcquisitionStrategy",
    "SEGO",
    "MFSEGO",
    "VFPI",
    "MOSEGO",
    "BiSEGO",
    "qPGP",
]
