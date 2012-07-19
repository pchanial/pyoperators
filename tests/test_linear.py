from __future__ import division

import numpy as np
import pyoperators

from pyoperators import Operator, BlockColumnOperator, DiagonalOperator
from pyoperators.linear import (DiagonalNumexprOperator,
                                DiagonalNumexprSeparableOperator,
                                IntegrationTrapezeWeightOperator,
                                PackOperator, UnpackOperator, SumOperator)
from pyoperators.utils.testing import (assert_eq, assert_is_instance,
                                       assert_is_none, assert_raises)
from .common import TestIdentityOperator, assert_inplace_outplace

SHAPES = (None, (), (1,), (3,), (2,3), (2,3,4))


def test_diagonal_numexpr():
    diag = np.array([1, 2, 3])
    expr = '(data+1)*3'
    def func1(cls, args, broadcast):
        assert_raises(ValueError, cls, broadcast=broadcast, *args)
    def func2(cls, args, broadcast, values):
        if broadcast == 'rightward':
            expected = (values.T*(diag.T+1)*3).T
        else:
            expected = values*(diag+1)*3
        op = cls(broadcast=broadcast, *args)
        if broadcast in ('leftward', 'rightward'):
            assert op.broadcast == broadcast
            assert_is_none(op.shapein)
        else:
            assert op.broadcast == 'disabled'
            assert_eq(op.shapein, diag.shape)
        assert_inplace_outplace(op, values, expected)
    for cls, args in zip((DiagonalNumexprOperator,
                          DiagonalNumexprSeparableOperator),
                         ((expr, {'data':diag}), (diag, expr))):
        for broadcast in (None, 'rightward', 'leftward', 'disabled'):
            if cls is DiagonalNumexprOperator and broadcast == 'rightward':
                yield func1, cls, args, broadcast
                continue
            for values in (np.array([3,2,1.]),
                           np.array([[1,2,3],[2,3,4],[3,4,5.]])):
                if values.ndim > 1 and broadcast in (None, 'disabled'):
                    continue
                yield func2, cls, args, broadcast, values

def test_diagonal_numexpr2():
    diag = np.array([1, 2, 3])
    d1 = DiagonalNumexprOperator('(data+1)*3', {'data':diag})
    d2 = DiagonalNumexprOperator('(data+2)*2', {'data':np.array([3,2,1])})
    d = d1 * d2
    assert_is_instance(d, DiagonalOperator)
    assert_eq(d.broadcast, 'disabled')
    assert_eq(d.shapein, (3,))
    assert_eq(d.data, [60, 72, 72])
    c = BlockColumnOperator(3*[TestIdentityOperator()], new_axisout=0)
    v = 2
    assert_inplace_outplace(d1*c, v, d1(c(v)))

def test_diagonal_numexpr3():
    d1 = DiagonalNumexprSeparableOperator([1,2,3], '(data+1)*3',
                                          broadcast='rightward')
    d2 = DiagonalNumexprSeparableOperator([3,2,1], '(data+2)*2')
    d = d1 * d2
    assert_is_instance(d, DiagonalOperator)
    assert_eq(d.broadcast, 'disabled')
    assert_eq(d.data, [60, 72, 72])
    c = BlockColumnOperator(3*[TestIdentityOperator()], new_axisout=0)
    v = [1,2]
    assert_inplace_outplace(d1*c, v, d1(c(v)))

def test_integration_trapeze():
    @pyoperators.decorators.square
    class Op(Operator):
        """ output[i] = value ** (i + input[i]) """
        def __init__(self, x):
            Operator.__init__(self, dtype=float)
            self.x = x
        def direct(self, input, output):
            output[...] = self.x ** (np.arange(input.size) + input)

    value = range(3)
    x = [0.5,1,2,4]
    func_op = BlockColumnOperator([Op(_) for _ in x], new_axisout=0)
    eval_ = func_op(value)
    expected = np.trapz(eval_, x=x, axis=0)
    integ = IntegrationTrapezeWeightOperator(x) * func_op
    assert_eq(integ(value), expected)
    

def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1,2,3,4,5]), [1,4])
    assert np.allclose(p.T([1,4]), [1,0,0,4,0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1,4]), [1,0,0,4,0])
    assert np.allclose(u.T([1,2,3,4,5]), [1,4])

def test_sum_operator():
    for s in SHAPES[2:]:
        for a in [None] + list(range(len(s))):
            op = SumOperator(axis=a)
            d = op.todense(shapein=s)
            t = op.T.todense(shapeout=s)
            assert_eq(d, t.T)
