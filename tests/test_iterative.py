"""
Testing of the iterative module

"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pyoperators
from pyoperators import IdentityOperator
from pyoperators.iterative.algorithms import acg
from pyoperators.iterative.cg import PCGAlgorithm, pcg
from pyoperators.utils.testing import assert_same

# collection of definite positive symmetric linear operators to test
OPERATORS = [
    pyoperators.DiagonalOperator(np.random.rand(16)),
    pyoperators.TridiagonalOperator(np.arange(1, 17), np.arange(1, 16)),
]

# collection of vectors
VECTORS = [np.ones(16), np.arange(1, 17)]

# collection of old solvers
METHODS = [acg]

# collection of solvers
CLASSES = [PCGAlgorithm]
SOLVERS = [pcg]


@pytest.mark.xfail(reason='reason: Unknown.')
@pytest.mark.parametrize('operator', OPERATORS)
@pytest.mark.parametrize('vector', VECTORS)
@pytest.mark.parametrize('method', METHODS)
def test_methods_inv(operator, vector, method):
    y = operator @ vector
    xe = method(operator, y, maxiter=100, tol=1e-7)
    assert_same(vector, xe)


@pytest.mark.parametrize('operator', OPERATORS)
@pytest.mark.parametrize('vector', VECTORS)
@pytest.mark.parametrize('cls', CLASSES)
def test_classes_inv(operator, vector, cls):
    y = operator(vector)
    algo = cls(operator, y, maxiter=100, tol=1e-7)
    xe = algo.run()
    assert_allclose(vector, xe, rtol=1e-5)


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('vector', VECTORS)
def test_solution_as_x0(solver, vector):
    solution = solver(IdentityOperator(shapein=vector.shape), vector, x0=vector)
    assert_same(solution['nit'], 0)
    assert_same(solution['x'], vector)
