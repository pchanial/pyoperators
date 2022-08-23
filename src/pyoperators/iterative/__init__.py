"""
Package defining algorithm working on Operators.
Contains the following modules:

- criterions: define criterions to be minimized by algorithms.

- optimize: a wrapper for scipy.optimize "fmin" functions.

- algorithms: defines iterative minimization algorithms working on criterions.

- dli: Defines the Lanczos algorithm and the DoubleLoopInference algorithm.
"""

# these modules are WIP
from .cg import pcg
from .core import AbnormalStopIteration, IterativeAlgorithm
from .stopconditions import (
    MaxErrorStopCondition,
    MaxIterationStopCondition,
    StopCondition,
)

__all__ = [
    'AbnormalStopIteration',
    'IterativeAlgorithm',
    'MaxErrorStopCondition',
    'MaxIterationStopCondition',
    'StopCondition',
    'pcg',
]
