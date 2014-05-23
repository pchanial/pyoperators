#!/usr/bin/env python

"""
Testing of the iterative module

"""

import numpy as np
import pyoperators
from numpy.testing import assert_allclose
from pyoperators import IdentityOperator, iterative
from pyoperators.utils.testing import assert_same, skiptest

# collection of definite positive symmetric linear operators to test
operator_list = [pyoperators.DiagonalOperator(np.random.rand(16)),
                 pyoperators.TridiagonalOperator(np.arange(1, 17),
                                                 np.arange(1, 16))]

# collection of vectors
vector_list = [np.ones(16), np.arange(1, 17)]

# collection of old solvers
methods = [iterative.algorithms.acg]

# collection of solvers
classes = [iterative.cg.PCGAlgorithm]
solvers = [iterative.cg.pcg]


@skiptest
def test_methods_inv():
    def func(m, A, x):
        y = A * x
        xe = m(A, y, maxiter=100, tol=1e-7)
        assert_same(x, xe)
    for A in operator_list:
        for x in vector_list:
            for m in methods:
                yield func, m, A, x


def test_classes_inv():
    def func(c, A, x):
        y = A(x)
        algo = c(A, y, maxiter=100, tol=1e-7)
        xe = algo.run()
        assert_allclose(x, xe, rtol=1e-6)
    for A in operator_list:
        for x in vector_list:
            for c in classes:
                yield func, c, A, x


def test_solution_as_x0():
    def func(s, v):
        solution = s(IdentityOperator(shapein=v.shape), v, x0=v)
        assert_same(solution['nit'], 0)
        assert_same(solution['x'], v)
    for s in solvers:
        for v in vector_list:
            yield func, s, v
