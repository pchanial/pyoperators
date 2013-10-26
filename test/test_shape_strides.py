from __future__ import division

import numpy as np
from nose import SkipTest
from numpy.testing import assert_equal, assert_raises
from pyoperators import Operator, IdentityOperator, ReshapeOperator, SwapAxesOperator
from pyoperators.decorators import contiguous, square
from pyoperators.utils import product
from pyoperators.utils.testing import assert_is, assert_same


def test_reshape():
    shapein = (2, 3, 4)
    shapeouts = (1, 2, 4, 3), (24,), (8, 3)
    input = np.arange(product(shapein)).reshape(shapein)

    def func(shapeout):
        op = ReshapeOperator(shapein, shapeout)
        assert_same(op(input), input.reshape(shapeout))
        dense = op.todense()
        assert_same(dense.T, op.T.todense())
        assert_same(np.dot(dense.T, dense), np.eye(24))
        dense = op.todense(inplace=True)
        assert_same(dense.T, op.T.todense(inplace=True))
        assert_same(np.dot(dense.T, dense), np.eye(24))

    for shapeout in shapeouts:
        yield func, shapeout


def test_reshape_composition():
    shape1 = (2, 3, 4)
    shape2 = (24,)
    shape3 = (4, 3, 2, 1)
    assert_equal(
        ReshapeOperator(shape2, shape3) * ReshapeOperator(shape1, shape2),
        ReshapeOperator(shape1, shape3),
    )


def test_reshape_identity():
    shape = (2, 3, 4)
    assert_equal(ReshapeOperator(shape, shape), IdentityOperator(shape))
    raise SkipTest()
    op = ReshapeOperator((2, 3, 4), (4, 3, 2))
    assert_is(op.I, op.T)


def test_reshape_errors():
    assert_raises(ValueError, ReshapeOperator, (2, 3, 4), None)
    assert_raises(ValueError, ReshapeOperator, None, (2, 3, 4))
    assert_raises(ValueError, ReshapeOperator, (3, 3, 4), (2, 3, 4))


def test_swapaxes():
    shapein = (2, 3, 4)
    axes = (-2, -1, 0, 1, 2)
    input = np.arange(product(shapein)).reshape(shapein)

    def func(a1, a2):
        op = SwapAxesOperator(a1, a2)
        assert_same(op(input), input.swapaxes(a1, a2))
        dense = op.todense(shapein=shapein)
        assert_same(dense.T, op.T.todense(shapeout=shapein))
        assert_same(np.dot(dense.T, dense), np.eye(24))
        dense = op.todense(inplace=True, shapein=shapein)
        assert_same(dense.T, op.T.todense(inplace=True, shapeout=shapein))
        assert_same(np.dot(dense.T, dense), np.eye(24))

    for a1 in axes:
        for a2 in axes:
            yield func, a1, a2


def test_swapaxes_composition():
    a1 = 1
    a2 = 2
    a3 = 3
    raise SkipTest()
    assert_equal(
        SwapAxesOperator(a2, a3) * SwapAxesOperator(a1, a2), SwapAxesOperator(a1, a3)
    )


def test_swapaxes_identity():
    assert_equal(SwapAxesOperator(1, 1), IdentityOperator())
    raise SkipTest()
    op = SwapAxesOperator(1, 2)
    assert_is(op.I, op.T)


def test_swapaxes_errors():
    input = np.arange(24).reshape((2, 3, 4))
    assert_raises(ValueError, SwapAxesOperator(1, 3), input)
    assert_raises(ValueError, SwapAxesOperator(3, 1), input)
    assert_raises(ValueError, SwapAxesOperator(1, -4), input)
    assert_raises(ValueError, SwapAxesOperator(-4, 1), input)
