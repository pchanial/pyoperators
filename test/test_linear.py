from __future__ import division

import numpy as np
import pyoperators

from numpy.testing import assert_allclose
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, DegreesOperator,
    DenseOperator, DiagonalOperator, DiagonalNumexprOperator,
    DifferenceOperator, IntegrationTrapezeOperator, Operator, RadiansOperator,
    Rotation2dOperator, Rotation3dOperator, TridiagonalOperator,
    SymmetricBandToeplitzOperator, SumOperator)
from pyoperators.utils import product
from pyoperators.utils.testing import (
    assert_eq, assert_is_instance, assert_is_none, assert_is_type,
    assert_same)
from .common import IdentityOutplaceOperator, assert_inplace_outplace

SHAPES = ((), (1,), (3,), (2, 3), (2, 3, 4))


def test_degrees():
    def func(dtype):
        d = DegreesOperator(dtype=dtype)
        assert_same(d(1), np.degrees(np.ones((), dtype=dtype)))
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        yield func, dtype


def test_degrees_rules():
    d = DegreesOperator()
    assert_is_type(d.I, RadiansOperator)


def test_diagonal_numexpr():
    diag = np.array([1, 2, 3])
    expr = '(data+1)*3'

    def func(broadcast, values):
        if broadcast == 'rightward':
            expected = (values.T*(diag.T+1)*3).T
        else:
            expected = values*(diag+1)*3
        op = DiagonalNumexprOperator(diag, expr, broadcast=broadcast)
        if broadcast in ('leftward', 'rightward'):
            assert op.broadcast == broadcast
            assert_is_none(op.shapein)
        else:
            assert op.broadcast == 'disabled'
            assert_eq(op.shapein, diag.shape)
            assert_eq(op.shapeout, diag.shape)
        assert_inplace_outplace(op, values, expected)
    for broadcast in (None, 'rightward', 'leftward', 'disabled'):
        for values in (np.array([3, 2, 1.]),
                       np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5.]])):
            if values.ndim > 1 and broadcast in (None, 'disabled'):
                continue
            yield func, broadcast, values


def test_diagonal_numexpr2():
    d1 = DiagonalNumexprOperator([1, 2, 3], '(data+1)*3',
                                 broadcast='rightward')
    d2 = DiagonalNumexprOperator([3, 2, 1], '(data+2)*2')
    d = d1 * d2
    assert_is_instance(d, DiagonalOperator)
    assert_eq(d.broadcast, 'disabled')
    assert_eq(d.data, [60, 72, 72])
    c = BlockColumnOperator(3*[IdentityOutplaceOperator()], new_axisout=0)
    v = [1, 2]
    assert_inplace_outplace(d1*c, v, d1(c(v)))


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
    @pyoperators.flags.square
    class Op(Operator):
        """ output[i] = value ** (i + input[i]) """
        def __init__(self, x):
            Operator.__init__(self, dtype=float)
            self.x = x

        def direct(self, input, output):
            output[...] = self.x ** (np.arange(input.size) + input)

    value = list(range(3))
    x = [0.5, 1, 2, 4]
    func_op = BlockColumnOperator([Op(_) for _ in x], new_axisout=0)
    eval_ = func_op(value)
    expected = np.trapz(eval_, x=x, axis=0)
    integ = IntegrationTrapezeOperator(x)(func_op)
    assert_same(integ(value), expected)


def test_radians():
    def func(dtype):
        d = RadiansOperator(dtype=dtype)
        assert_same(d(1), np.radians(np.ones((), dtype=dtype)))
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        yield func, dtype


def test_radians_rules():
    d = RadiansOperator()
    assert_is_type(d.I, DegreesOperator)


def test_rotation_2d():
    def func(shape, degrees):
        angle = np.arange(product(shape)).reshape(shape)
        if degrees:
            angle_ = np.radians(angle)
        else:
            angle_ = angle
        angle_ = angle_.reshape(angle.size)
        r = Rotation2dOperator(angle, degrees=degrees)
        actual = r([1, 0]).reshape((angle.size, 2))
        expected = np.array([np.cos(angle_), np.sin(angle_)]).T
        assert_same(actual, expected)
    for shape in SHAPES:
        for degrees in False, True:
            yield func, shape, degrees


def test_rotation_3d_1axis():
    rx = Rotation3dOperator('X', 90, degrees=True)
    ry = Rotation3dOperator('Y', 90, degrees=True)
    rz = Rotation3dOperator('Z', 90, degrees=True)
    ref = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

    # single axis rotations
    exps = (
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    def func(rot, exp):
        assert_allclose(rot(ref), exp, atol=1e-15)
    for rot, exp in zip((rx, ry, rz), exps):
        yield func, rot, exp


def test_rotation_3d_2axis():
    ref = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
    alpha = 0.1
    beta = 0.2

    # intrinsic rotations
    conventions = ("XY'", "XZ'", "YX'", "YZ'", "ZX'", "ZY'")

    def func(c):
        r = Rotation3dOperator(c, alpha, beta)
        r2 = Rotation3dOperator(c[0], alpha) * \
             Rotation3dOperator(c[1], beta)
        assert_allclose(r(ref), r2(ref))
    for c in conventions:
        yield func, c

    # extrinsic rotations
    conventions = ('XY', 'XZ', 'YX', 'YZ', 'ZX', 'ZY')

    def func(c):
        r = Rotation3dOperator(c, alpha, beta)
        r2 = Rotation3dOperator(c[1], beta) * \
             Rotation3dOperator(c[0], alpha)
        assert_allclose(r(ref), r2(ref))
    for c in conventions:
        yield func, c


def test_rotation_3d_3axis():
    ref = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
    alpha = 0.1
    beta = 0.2
    gamma = 0.3

    # intrinsic rotations
    conventions = ("XZ'X''", "XZ'Y''",
                   "XY'X''", "XY'Z''",
                   "YX'Y''", "YX'Z''",
                   "YZ'Y''", "YZ'X''",
                   "ZY'Z''", "ZY'X''",
                   "ZX'Z''", "ZX'Y''")

    def func(c):
        r = Rotation3dOperator(c, alpha, beta, gamma)
        r2 = Rotation3dOperator(c[0], alpha) * \
             Rotation3dOperator(c[1], beta) * \
             Rotation3dOperator(c[3], gamma)
        assert_allclose(r(ref), r2(ref))
    for c in conventions:
        yield func, c

    # extrinsic rotations
    conventions = ("XZX", "XZY",
                   "XYX", "XYZ",
                   "YXY", "YXZ",
                   "YZY", "YZX",
                   "ZYZ", "ZYX",
                   "ZXZ", "ZXY")

    def func(c):
        r = Rotation3dOperator(c, alpha, beta, gamma)
        r2 = Rotation3dOperator(c[2], gamma) * \
             Rotation3dOperator(c[1], beta) * \
             Rotation3dOperator(c[0], alpha)
        assert_allclose(r(ref), r2(ref))
    for c in conventions:
        yield func, c


def test_sum_operator():
    for s in SHAPES[1:]:
        for a in [None] + list(range(len(s))):
            op = SumOperator(axis=a)
            d = op.todense(shapein=s)
            t = op.T.todense(shapeout=s)
            assert_eq(d, t.T)


def test_symmetric_band_toeplitz_operator():
    def totoeplitz(n, firstrow):
        if isinstance(n, tuple):
            n_ = n[-1]
            return BlockDiagonalOperator(
                [totoeplitz(n_, f_) for f_ in firstrow], new_axisin=0)
        ncorr = len(firstrow) - 1
        dense = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i-j) <= ncorr:
                    dense[i, j] = firstrow[abs(i-j)]
        return DenseOperator(dense, shapein=dense.shape[1])

    def func(n, firstrow):
        s = SymmetricBandToeplitzOperator(n, firstrow)
        if firstrow == [1] or firstrow == [[2], [1]]:
            assert_is_instance(s, DiagonalOperator)
        assert_same(s.todense(), totoeplitz(n, firstrow).todense(), atol=1)

    for n in [2, 3, 4, 5]:
        for firstrow in ([1], [2, 1]):
            yield func, n, firstrow
    for n in ((2, _) for _ in [2, 3, 4, 5]):
        for firstrow in ([[2], [1]], [[2, 1], [3, 2]]):
            yield func, n, firstrow


def test_tridiagonal_operator():
    values = (
        ([1, 1, 0], [2, 1], [2, 2]),
        ([1, 1, 2], [2, 1], None),
        ([1j, 1, 0], [2, 1], [-1j, 2]),
        ([1, 1j, 2], [2j, 1], None))
    expected = ([[1, 2, 0],
                 [2, 1, 2],
                 [0, 1, 0]],
                [[1, 2, 0],
                 [2, 1, 1],
                 [0, 1, 2]],
                [[1j,-1j, 0],
                 [ 2,  1, 2],
                 [ 0,  1, 0]],
                [[ 1,-2j, 0],
                 [2j, 1j, 1],
                 [ 0,  1, 2]])

    def func(v, e):
        o = TridiagonalOperator(v[0], v[1], v[2])
        assert_eq(o.todense(), e)
        assert_eq(o.T.todense(), e.T)
        assert_eq(o.C.todense(), e.conj())
        assert_eq(o.H.todense(), e.T.conj())
    for v, e in zip(values, expected):
        yield func, v, np.array(e)
