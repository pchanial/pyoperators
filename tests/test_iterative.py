#!/usr/bin/env python

"""
Testing of the iterative module
"""

import nose
from numpy import testing
import numpy as np
import operators
from operators import iterative

# collection of linear operators to test
operator_list = [
    operators.DiagonalOperator(np.random.rand(16)),
    operators.DiagonalOperator(np.arange(1, 17)),
]

# collection of vectors
vector_list = [np.ones(16), np.arange(1, 17), np.random.rand(16)]

# collection of methods
methods = [
    iterative.acg,
]

# tests
def check_inv(method, A, x):
    y = A * x
    xe = method(A, y, maxiter=100, tol=1e-6)
    testing.assert_almost_equal, x, xe


def test_inv():
    for A in operator_list:
        for x in vector_list:
            for m in methods:
                yield check_inv, m, A, x


if __name__ == "__main__":
    nose.run(argv=['', __file__])
