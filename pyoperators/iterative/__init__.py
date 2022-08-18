"""
Package defining algorithm working on Operators.
Contains the following modules:

- criterions: define criterions to be minimized by algorithms.

- optimize: a wrapper for scipy.optimize "fmin" functions.

- algorithms: defines iterative minimization algorithms working on criterions.

- dli: Defines the Lanczos algorithm and the DoubleLoopInference algorithm.
"""

# these modules are WIP
from . import algorithms, criterions, dli, optimize
from .cg import *
from .core import *
from .stopconditions import *

del core
