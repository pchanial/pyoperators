"""
The pyoperator package contains the following modules or packages:

- core : defines the Operator class
- linear : defines standard linear operators
- nonlinear : defines non-linear operators (such as thresholding or rounding)
- iterative : defines iterative algorithms working with operators

- operators_mpi : (optional) loaded if mpi4py is present.
- operators_pywt : (optional) loaded if PyWavelets is present.
"""

from .utils import *
from .core import *
from .linear import *
from .nonlinear import *
from .iterative import *

try:
    from .operators_pywt import *
except (ImportError):
    pass

try:
    from .operators_mpi import *
except (ImportError):
    pass

__version__ = '0.4'

import types

__all__ = [
    f for f in dir() if f[0] != '_' and not isinstance(eval(f), types.ModuleType)
]
del types
