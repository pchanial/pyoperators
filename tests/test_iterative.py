#!/usr/bin/env python

"""
Testing of the iterative module
"""

from numpy import testing
import numpy as np
import pyoperators
from pyoperators import iterative

# collection of definite positive symmetric linear operators to test
operator_list = [pyoperators.DiagonalOperator(np.random.rand(16)),
                 pyoperators.TridiagonalOperator(np.arange(1,17),
                                                 np.arange(1,16)),
                 ]

# collection of vectors
vector_list = [np.ones(16), np.arange(1, 17), np.random.rand(16)]

# collection of methods
methods = [iterative.algorithms.acg]

# collection of classes
classes = [iterative.cg.PCGAlgorithm]

def test_methods_inv():
    def func(m, A, x):
        y = A * x
        xe = m(A, y, maxiter=100, tol=1e-6)
        testing.assert_almost_equal, x, xe
    for A in operator_list:
        for x in vector_list:
            for m in methods:
                yield func, m, A, x

def test_classes_inv():
    def func(c, A, x):
        y = A(x)
        algo = c(A, y, maxiter=100, tol=1e-6)
        xe = algo.run()
        testing.assert_almost_equal, x, xe
    for A in operator_list:
        for x in vector_list:
            for c in classes:
                yield func, c, A, x
