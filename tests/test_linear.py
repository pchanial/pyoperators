from __future__ import division

import numpy as np
import pyoperators

from pyoperators import Operator, BlockColumnOperator, DiagonalOperator
from pyoperators.linear import (
    DiagonalNumexprNonSeparableOperator,
    DiagonalNumexprOperator,
    DifferenceOperator,
    IntegrationTrapezeWeightOperator,
    PackOperator,
    TridiagonalOperator,
    UnpackOperator,
    SumOperator,
)
from pyoperators.utils import product
from pyoperators.utils.testing import (
    assert_eq,
    assert_is_instance,
    assert_is_none,
    assert_raises,
)
from .common import IdentityOutplaceOperator, assert_inplace_outplace

SHAPES = (None, (), (1,), (3,), (2, 3), (2, 3, 4))


def test_diagonal_numexpr():
    diag = np.array([1, 2, 3])
    expr = '(data+1)*3'

    def func1(cls, args, broadcast):
        assert_raises(ValueError, cls, broadcast=broadcast, *args)

    def func2(cls, args, broadcast, values):
        if broadcast == 'rightward':
            expected = (values.T * (diag.T + 1) * 3).T
        else:
            expected = values * (diag + 1) * 3
        op = cls(broadcast=broadcast, *args)
        if broadcast in ('leftward', 'rightward'):
            assert op.broadcast == broadcast
            assert_is_none(op.shapein)
        else:
            assert op.broadcast == 'disabled'
            assert_eq(op.shapein, diag.shape)
        assert_inplace_outplace(op, values, expected)

    for cls, args in zip(
        (DiagonalNumexprNonSeparableOperator, DiagonalNumexprOperator),
        ((expr, {'data': diag}), (diag, expr)),
    ):
        for broadcast in (None, 'rightward', 'leftward', 'disabled'):
            if cls is DiagonalNumexprNonSeparableOperator and broadcast == 'rightward':
                yield func1, cls, args, broadcast
                continue
            for values in (
                np.array([3, 2, 1.0]),
                np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5.0]]),
            ):
                if values.ndim > 1 and broadcast in (None, 'disabled'):
                    continue
                yield func2, cls, args, broadcast, values


def test_diagonal_numexpr2():
    diag = np.array([1, 2, 3])
    d1 = DiagonalNumexprNonSeparableOperator('(data+1)*3', {'data': diag})
    d2 = DiagonalNumexprNonSeparableOperator(
        '(data+2)*2', {'data': np.array([3, 2, 1])}
    )
    d = d1 * d2
    assert_is_instance(d, DiagonalOperator)
    assert_eq(d.broadcast, 'disabled')
    assert_eq(d.shapein, (3,))
    assert_eq(d.data, [60, 72, 72])
    c = BlockColumnOperator(3 * [IdentityOutplaceOperator()], new_axisout=0)
    v = 2
    assert_inplace_outplace(d1 * c, v, d1(c(v)))


def test_diagonal_numexpr3():
    d1 = DiagonalNumexprOperator([1, 2, 3], '(data+1)*3', broadcast='rightward')
    d2 = DiagonalNumexprOperator([3, 2, 1], '(data+2)*2')
    d = d1 * d2
    assert_is_instance(d, DiagonalOperator)
    assert_eq(d.broadcast, 'disabled')
    assert_eq(d.data, [60, 72, 72])
    c = BlockColumnOperator(3 * [IdentityOutplaceOperator()], new_axisout=0)
    v = [1, 2]
    assert_inplace_outplace(d1 * c, v, d1(c(v)))


def test_diff_non_optimised():
    def func(shape, axis):
        dX = DifferenceOperator(axis=axis, shapein=shape)
        a = np.arange(product(shape)).reshape(shape)
        assert_eq(dX(a), np.diff(a, axis=axis))
        dX_dense = dX.todense()

        dXT_dense = dX.T.todense()
        assert_eq(dX_dense.T, dXT_dense)

    for shape in ((3,), (3, 4), (3, 4, 5), (3, 4, 5, 6)):
        for axis in range(len(shape)):
            yield func, shape, axis


def test_integration_trapeze():
    @pyoperators.decorators.square
    class Op(Operator):
        """output[i] = value ** (i + input[i])"""

        def __init__(self, x):
            Operator.__init__(self, dtype=float)
            self.x = x

        def direct(self, input, output):
            output[...] = self.x ** (np.arange(input.size) + input)

    value = range(3)
    x = [0.5, 1, 2, 4]
    func_op = BlockColumnOperator([Op(_) for _ in x], new_axisout=0)
    eval_ = func_op(value)
    expected = np.trapz(eval_, x=x, axis=0)
    integ = IntegrationTrapezeWeightOperator(x) * func_op
    assert_eq(integ(value), expected)


def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1, 2, 3, 4, 5]), [1, 4])
    assert np.allclose(p.T([1, 4]), [1, 0, 0, 4, 0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1, 4]), [1, 0, 0, 4, 0])
    assert np.allclose(u.T([1, 2, 3, 4, 5]), [1, 4])


def test_sum_operator():
    for s in SHAPES[2:]:
        for a in [None] + list(range(len(s))):
            op = SumOperator(axis=a)
            d = op.todense(shapein=s)
            t = op.T.todense(shapeout=s)
            assert_eq(d, t.T)


def test_tridiagonal_operator():
    values = (
        ([1, 1, 0], [2, 1], [2, 2]),
        ([1, 1, 2], [2, 1], None),
        ([1j, 1, 0], [2, 1], [-1j, 2]),
        ([1, 1j, 2], [2j, 1], None),
    )
    expected = (
        [[1, 2, 0], [2, 1, 2], [0, 1, 0]],
        [[1, 2, 0], [2, 1, 1], [0, 1, 2]],
        [[1j, -1j, 0], [2, 1, 2], [0, 1, 0]],
        [[1, -2j, 0], [2j, 1j, 1], [0, 1, 2]],
    )

    def func(v, e):
        o = TridiagonalOperator(v[0], v[1], v[2])
        assert_eq(o.todense(), e)
        assert_eq(o.T.todense(), e.T)
        assert_eq(o.C.todense(), e.conj())
        assert_eq(o.H.todense(), e.T.conj())

    for v, e in zip(values, expected):
        yield func, v, np.array(e)
