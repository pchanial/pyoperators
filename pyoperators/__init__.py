"""
The pyoperator package contains the following modules or packages:

- core : defines the Operator class
- linear : defines standard linear operators
- nonlinear : defines non-linear operators (such as thresholding or rounding)
- iterative : defines iterative algorithms working with operators

- pywt_operators : (optional) loaded if PyWavelets is present.
"""

from .core import *
from .linear import *
from .nonlinear import *
from .iterative import *

# try to import pywt
try:
    from .pywt_operators import *
except(ImportError):
    pass

__version__ = '0.1'
