import sys

import numpy as np
import pytest

try:
    import pyfftw  # noqa
except ImportError:
    pass
from numpy.testing import assert_allclose, assert_equal

import pyoperators
from pyoperators import (
    BlockColumnOperator,
    BlockDiagonalOperator,
    DegreesOperator,
    DenseOperator,
    DiagonalNumexprOperator,
    DiagonalOperator,
    DifferenceOperator,
    IntegrationTrapezeOperator,
    Operator,
    RadiansOperator,
    Rotation2dOperator,
    Rotation3dOperator,
    SumOperator,
    SymmetricBandToeplitzOperator,
    TridiagonalOperator,
)
from pyoperators.utils import product
from pyoperators.utils.testing import assert_same

from .common import FLOAT_DTYPES, IdentityOutplace

SHAPES = ((), (1,), (3,), (2, 3), (2, 3, 4))


def assert_inplace_outplace(op, v, expected):
    w = op(v)
    assert_equal(w, expected)
    op(v, out=w)
    assert_equal(w, expected)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_degrees(dtype):
    d = DegreesOperator(dtype=dtype)
    assert_same(d(1), np.degrees(np.ones((), dtype=dtype)))


def test_degrees_rules():
    d = DegreesOperator()
    assert type(d.I) is RadiansOperator


@pytest.mark.parametrize('broadcast', [None, 'rightward', 'leftward', 'disabled'])
@pytest.mark.parametrize(
    'values', [np.array([3, 2, 1.0]), np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5.0]])]
)
def test_diagonal_numexpr(broadcast, values):
    if values.ndim > 1 and broadcast in (None, 'disabled'):
        return

    diag = np.array([1, 2, 3])
    expr = '(data + 1) * 3'

    if broadcast == 'rightward':
        expected = (values.T * (diag.T + 1) * 3).T
    else:
        expected = values * (diag + 1) * 3
    op = DiagonalNumexprOperator(diag, expr, broadcast=broadcast)

    if broadcast in ('leftward', 'rightward'):
        assert op.broadcast == broadcast
        assert op.shapein is None
    else:
        assert op.broadcast == 'disabled'
        assert op.shapein == diag.shape
        assert op.shapeout == diag.shape

    assert_inplace_outplace(op, values, expected)


def test_diagonal_numexpr2():
    d1 = DiagonalNumexprOperator([1, 2, 3], '(data + 1) * 3', broadcast='rightward')
    d2 = DiagonalNumexprOperator([3, 2, 1], '(data + 2) * 2')
    d = d1 @ d2
    assert isinstance(d, DiagonalOperator)
    assert d.broadcast == 'disabled'
    assert_equal(d.data, [60, 72, 72])
    c = BlockColumnOperator(3 * [IdentityOutplace()], new_axisout=0)
    v = [1, 2]
    assert_inplace_outplace(d1 @ c, v, d1(c(v)))


@pytest.mark.parametrize('shape', [(3,), (3, 4), (3, 4, 5), (3, 4, 5, 6)])
def test_diff_non_optimised(shape):
    for axis in range(len(shape)):
        dX = DifferenceOperator(axis=axis, shapein=shape)
        a = np.arange(product(shape)).reshape(shape)
        assert_equal(dX(a), np.diff(a, axis=axis))
        dX_dense = dX.todense()

        dXT_dense = dX.T.todense()
        assert_equal(dX_dense.T, dXT_dense)


def test_integration_trapeze():
    @pyoperators.flags.square
    class Op(Operator):
        """output[i] = value ** (i + input[i])"""

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


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_radians(dtype):
    d = RadiansOperator(dtype=dtype)
    assert_same(d(1), np.radians(np.ones((), dtype=dtype)))


def test_radians_rules():
    d = RadiansOperator()
    assert type(d.I) is DegreesOperator


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('degrees', [False, True])
def test_rotation_2d(shape, degrees):
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


@pytest.mark.parametrize(
    'convention, expected',
    [
        ('X', [[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        ('Y', [[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        ('Z', [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    ],
)
def test_rotation_3d_1axis(convention, expected):
    rot = Rotation3dOperator(convention, 90, degrees=True)
    ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert_allclose(rot(ref), expected, atol=1e-15)


@pytest.mark.parametrize('convention', ["XY'", "XZ'", "YX'", "YZ'", "ZX'", "ZY'"])
def test_intrinsic_rotation_3d_2axis(convention):
    ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    alpha = 0.1
    beta = 0.2
    r = Rotation3dOperator(convention, alpha, beta)
    r2 = Rotation3dOperator(convention[0], alpha) @ Rotation3dOperator(
        convention[1], beta
    )
    assert_allclose(r(ref), r2(ref))


@pytest.mark.parametrize('convention', ['XY', 'XZ', 'YX', 'YZ', 'ZX', 'ZY'])
def test_extrinsic_rotation_3d_2axis(convention):
    ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    alpha = 0.1
    beta = 0.2
    r = Rotation3dOperator(convention, alpha, beta)
    r2 = Rotation3dOperator(convention[1], beta) @ Rotation3dOperator(
        convention[0], alpha
    )
    assert_allclose(r(ref), r2(ref))


@pytest.mark.parametrize(
    'convention',
    [
        "XZ'X''",
        "XZ'Y''",
        "XY'X''",
        "XY'Z''",
        "YX'Y''",
        "YX'Z''",
        "YZ'Y''",
        "YZ'X''",
        "ZY'Z''",
        "ZY'X''",
        "ZX'Z''",
        "ZX'Y''",
    ],
)
def test_intrinsic_rotation_3d_3axis(convention):
    ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    alpha = 0.1
    beta = 0.2
    gamma = 0.3
    r = Rotation3dOperator(convention, alpha, beta, gamma)
    r2 = (
        Rotation3dOperator(convention[0], alpha)
        @ Rotation3dOperator(convention[1], beta)
        @ Rotation3dOperator(convention[3], gamma)
    )
    assert_allclose(r(ref), r2(ref))


@pytest.mark.parametrize(
    'convention',
    [
        'XZX',
        'XZY',
        'XYX',
        'XYZ',
        'YXY',
        'YXZ',
        'YZY',
        'YZX',
        'ZYZ',
        'ZYX',
        'ZXZ',
        'ZXY',
    ],
)
def test_extrinsic_rotation_3d_3axis(convention):
    ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    alpha = 0.1
    beta = 0.2
    gamma = 0.3
    r = Rotation3dOperator(convention, alpha, beta, gamma)
    r2 = (
        Rotation3dOperator(convention[2], gamma)
        @ Rotation3dOperator(convention[1], beta)
        @ Rotation3dOperator(convention[0], alpha)
    )
    assert_allclose(r(ref), r2(ref))


@pytest.mark.parametrize('shape', SHAPES[1:])
def test_sum_operator(shape):
    for axis in [None, *range(len(shape))]:
        op = SumOperator(axis=axis)
        d = op.todense(shapein=shape)
        t = op.T.todense(shapeout=shape)
        assert_equal(d, t.T)


def assert_symmetric_band_toeplitz(n, firstrow):
    def totoeplitz(n, firstrow):
        if isinstance(n, tuple):
            n_ = n[-1]
            return BlockDiagonalOperator(
                [totoeplitz(n_, f_) for f_ in firstrow], new_axisin=0
            )
        ncorr = len(firstrow) - 1
        dense = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= ncorr:
                    dense[i, j] = firstrow[abs(i - j)]
        return DenseOperator(dense, shapein=dense.shape[1])

    s = SymmetricBandToeplitzOperator(n, firstrow)
    if firstrow == [1] or firstrow == [[2], [1]]:
        assert isinstance(s, DiagonalOperator)
    assert_same(s.todense(), totoeplitz(n, firstrow).todense(), atol=1)


@pytest.mark.skipif(
    'pyfftw' not in sys.modules, reason='SymmetricBandToeplitzOperator requires pyfftw.'
)
@pytest.mark.parametrize('n', [2, 3, 4, 5])
@pytest.mark.parametrize('firstrow', [[1], [2, 1]])
def test_symmetric_band_toeplitz_operator(n, firstrow):
    assert_symmetric_band_toeplitz(n, firstrow)


@pytest.mark.skipif(
    'pyfftw' not in sys.modules, reason='SymmetricBandToeplitzOperator requires pyfftw.'
)
@pytest.mark.parametrize('n', [(2, _) for _ in [2, 3, 4, 5]])
@pytest.mark.parametrize('firstrow', [[[2], [1]], [[2, 1], [3, 2]]])
def test_block_symmetric_band_toeplitz_operator(n, firstrow):
    assert_symmetric_band_toeplitz(n, firstrow)


@pytest.mark.parametrize(
    'diagonals, expected',
    [
        (([1, 1, 0], [2, 1], [2, 2]), [[1, 2, 0], [2, 1, 2], [0, 1, 0]]),
        (([1, 1, 2], [2, 1], None), [[1, 2, 0], [2, 1, 1], [0, 1, 2]]),
        (([1j, 1, 0], [2, 1], [-1j, 2]), [[1j, -1j, 0], [2, 1, 2], [0, 1, 0]]),
        (([1, 1j, 2], [2j, 1], None), [[1, -2j, 0], [2j, 1j, 1], [0, 1, 2]]),
    ],
)
def test_tridiagonal_operator(diagonals, expected):
    expected = np.array(expected)
    op = TridiagonalOperator(*diagonals)
    assert_equal(op.todense(), expected)
    assert_equal(op.T.todense(), expected.T)
    assert_equal(op.C.todense(), expected.conj())
    assert_equal(op.H.todense(), expected.T.conj())
