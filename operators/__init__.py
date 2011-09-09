"""
The operators contains the following packages:

core : defines the Operator class
linear : defines standard linear operators (can be seen as matrices)
nonlinear : defines non-linear operators (such as thresholding or rounding)
iterative : defines iterative algorithms working with operators

Optionnally, if PyWavelets ( https://github.com/nigma/pywt.git ) is
present, the pywt_operators is loaded.
"""

from .core import *
from .linear import *
from .nonlinear import *
from .iterative import *

# try to import pywt
try:
    import pywt
except(ImportError):
    pass

if "pywt" in locals():
    from .pywt_operators import *
