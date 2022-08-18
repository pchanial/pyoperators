"""
The PyOperators package contains the following modules or packages:

- core : defines the Operator class
- linear : defines standard linear operators
- nonlinear : defines non-linear operators (such as thresholding or rounding)
- iterative : defines iterative algorithms working with operators
- utils : miscellaneous routines
- operators_mpi : MPI operators (even if mpi4py is not present)
- operators_pywt : (optional) loaded if PyWavelets is present.

"""
import sys
import types
from importlib.metadata import version as _version

from . import iterative
from .core import *
from .fft import *
from .iterative import pcg
from .linear import *
from .nonlinear import *
from .operators_mpi import *
from .operators_pywt import *
from .proxy import *
from .rules import rule_manager
from .utils import *
from .utils.mpi import MPI
from .warnings import PyOperatorsWarning

__all__ = [
    f for f in dir() if f[0] != '_' and not isinstance(eval(f), types.ModuleType)
]

del sys
del types

I = IdentityOperator()
O = ZeroOperator()
X = Variable('X')

__version__ = _version('pyoperators')
