"""
Package defining algorithm working on Operators.
Contains the following modules:

- criterions: define criterions to be minimized by algorithms.

- optimize: a wrapper for scipy.optimize "fmin" functions.

- algorithms: defines iterative minimization algorithms working on criterions.

- dli: Defines the Lanczos algorithm and the DoubleLoopInference algorithm.
"""

from .criterions import *
from .optimize import *
from .algorithms import *
from .dli import *
