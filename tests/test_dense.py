import numpy as np
import pytest

from pyoperators import (
    BlockDiagonalOperator,
    DenseBlockDiagonalOperator,
    DenseOperator,
    HomothetyOperator,
    PyOperatorsWarning,
)
from pyoperators.rules import rule_manager
from pyoperators.utils import broadcast_shapes, product, reshape_broadcast
from pyoperators.utils.testing import assert_same


def assert_dense_matvec(matrix, op, vec):
    expected = matrix @ vec
    assert_same(op(vec), expected)
    if op.flags.square:
        w = vec.copy()
        op(w, w)
        assert_same(w, expected)


@pytest.mark.parametrize('vec', [np.array([1 + 0j, 0]), np.array([0 + 0j, 1])])
def test_dense_square(vec):
    m = np.array([[1, 1j], [2, 2]])
    d = DenseOperator(m)
    assert_dense_matvec(m, d, vec)
    assert_dense_matvec(m.conj(), d.C, vec)
    assert_dense_matvec(m.T, d.T, vec)
    assert_dense_matvec(m.T.conj(), d.H, vec)


#    mI = np.linalg.inv(m)
#    assert_dense_matvec(mI, d.I, vec)
#    assert_dense_matvec(mI.conj(), d.I.C, vec)
#    assert_dense_matvec(mI.T, d.I.T, vec)
#    assert_dense_matvec(mI.T.conj(), d.I.H, vec)


@pytest.mark.parametrize('vec', [np.array([1 + 0j, 0]), np.array([0 + 0j, 1])])
def test_dense_non_square1(vec):
    m = np.array([[1, 2], [3, 4j], [5, 6]])
    d = DenseOperator(m)
    assert_dense_matvec(m, d, vec)
    assert_dense_matvec(m.conj(), d.C, vec)


@pytest.mark.parametrize(
    'vec', [np.array([1 + 0j, 0, 0]), np.array([0j, 1, 0]), np.array([0j, 0, 1])]
)
def test_dense_non_square2(vec):
    m = np.array([[1, 2], [3, 4j], [5, 6]])
    d = DenseOperator(m)
    assert_dense_matvec(m.T, d.T, vec)
    assert_dense_matvec(m.T.conj(), d.T.C, vec)


@pytest.mark.parametrize('shapein', [(2,), (3, 2), (3, 1, 2)])
@pytest.mark.parametrize('shapeout', [(3,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize('extrainput', [(), (5,), (3, 4)])
def test_dense_shapes(shapein, shapeout, extrainput):
    datashape = shapeout + shapein
    inputshape = extrainput + shapein
    d = np.arange(product(datashape)).reshape(datashape)
    b = DenseOperator(
        d, naxesin=len(shapein), naxesout=len(shapeout), shapein=inputshape
    )
    bdense = b.todense()
    n = product(extrainput)
    d_ = d.reshape((product(shapeout), product(shapein)))
    expected = BlockDiagonalOperator(n * [d_], axisin=0).todense()
    assert_same(bdense, expected)


@pytest.mark.parametrize('shape', [(2,), (3, 2)])
def test_dense_error(shape):
    data = np.arange(product(shape)).reshape(shape)
    b = DenseOperator(data)
    with pytest.raises(ValueError):
        b(np.ones(3))


def test_dense_rule_homothety():
    m = np.array([[1, 2], [3, 4], [5, 6]])
    d = HomothetyOperator(2) @ DenseOperator(m)
    assert type(d) is DenseOperator
    assert_same(d.data, m * 2)
    d = HomothetyOperator(2j) @ DenseOperator(m)
    assert type(d) is DenseOperator
    assert_same(d.data, m * 2j)
    assert d.dtype == complex


@pytest.mark.parametrize('sn', [(2,), (3, 2)])
@pytest.mark.parametrize('sm', [(3,), (2, 3)])
@pytest.mark.parametrize('sl_data', [(), (4,), (1, 4), (2, 1), (2, 4)])
@pytest.mark.parametrize('sl_input', [(), (4,), (2, 4), (2, 1)])
@pytest.mark.parametrize('broadcast', ['leftward', 'rightward'])
def test_block_diagonal(sn, sm, sl_data, sl_input, broadcast):
    if len(sl_input) > len(sl_data):
        return
    datashape = sl_data + sm + sn
    d = np.arange(product(datashape)).reshape(datashape)
    b = DenseBlockDiagonalOperator(
        d, naxesin=len(sn), naxesout=len(sm), broadcast=broadcast
    )
    sl = broadcast_shapes(sl_data, sl_input)
    bdense = b.todense(shapein=sl + sn)
    d_ = reshape_broadcast(d, sl + sm + sn)
    d_ = d_.reshape(-1, product(sm), product(sn))
    expected = BlockDiagonalOperator([_ for _ in d_], axisin=0).todense(
        shapein=product(sl + sn)
    )
    assert_same(bdense, expected)
    bTdense = b.T.todense(shapein=sl + sm)
    assert_same(bTdense, expected.T)


@pytest.mark.parametrize('sn', [(2,), (3, 2)])
@pytest.mark.parametrize('sm', [(3,), (2, 3)])
@pytest.mark.parametrize('sl_data', [(), (4,), (1, 4), (2, 1), (2, 4)])
@pytest.mark.parametrize('sl_input', [(), (4,), (2, 4), (2, 1)])
@pytest.mark.parametrize('sp_input', [(1,), (3,), (2, 2)])
def test_block_diagonal_leftward(sn, sm, sl_data, sl_input, sp_input):
    # (L, M, N) @ (P, L', N) -> (P, L'', M)
    if len(sl_input) > len(sl_data):
        return
    datashape = sl_data + sm + sn
    d = np.arange(product(datashape)).reshape(datashape)
    b = DenseBlockDiagonalOperator(
        d, naxesin=len(sn), naxesout=len(sm), broadcast='leftward'
    )
    sl = broadcast_shapes(sl_data, sl_input)
    bdense = b.todense(shapein=sp_input + sl + sn)
    d_ = reshape_broadcast(d, sp_input + sl + sm + sn)
    d_ = d_.reshape(-1, product(sm), product(sn))
    expected = BlockDiagonalOperator([_ for _ in d_], axisin=0).todense(
        shapein=product(sp_input + sl + sn)
    )
    assert_same(bdense, expected)
    bTdense = b.T.todense(shapein=sp_input + sl + sm)
    assert_same(bTdense, expected.T)


@pytest.mark.parametrize('sn', [(2,), (3, 2)])
@pytest.mark.parametrize('sm', [(3,), (2, 3)])
@pytest.mark.parametrize('sl_data', [(), (4,), (1, 4), (2, 1), (2, 4)])
@pytest.mark.parametrize('sl_input', [(), (4,), (2, 4), (2, 1)])
@pytest.mark.parametrize('sp_input', [(1,), (3,), (2, 2)])
def test_block_diagonal_rightward(sn, sm, sl_data, sl_input, sp_input):
    # (L, M, N) @ (L', N, P) -> (L'', M, P)
    if len(sl_input) != len(sl_data):
        return
    datashape = sl_data + sm + sn
    d = np.arange(product(datashape)).reshape(datashape)
    b = DenseBlockDiagonalOperator(
        d, naxesin=len(sn), naxesout=len(sm), broadcast='rightward'
    )
    sl = broadcast_shapes(sl_data, sl_input)
    bdense = b.todense(shapein=sl + sn + sp_input)
    d_ = reshape_broadcast(d, sl + sm + sn)
    d_ = d_.reshape(-1, product(sm), product(sn))
    expected = np.kron(
        BlockDiagonalOperator(list(d_), axisin=0).todense(shapein=product(sl + sn)),
        np.eye(product(sp_input)),
    )
    assert_same(bdense, expected)
    bTdense = b.T.todense(shapein=sl + sm + sp_input)
    assert_same(bTdense, expected.T)


@pytest.mark.parametrize('cls', [DenseBlockDiagonalOperator, DenseOperator])
def test_morphing1(cls):
    d = cls(3.0)
    assert type(d) is HomothetyOperator


@pytest.mark.parametrize('shape', [(3,), (1, 3), (2, 3)])
def test_morphing2(shape):
    d = DenseBlockDiagonalOperator(np.ones(shape))
    assert type(d) is DenseOperator


def test_warning():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    a = a.swapaxes(0, 1)
    with pytest.warns(PyOperatorsWarning):
        DenseOperator(a, naxesin=2)


@pytest.mark.parametrize(
    'shape1, shape2',
    [
        ((), (3,)),
        ((3,), ()),
        ((3,), (3,)),
        ((3,), (1,)),
        ((1,), (3,)),
        ((1, 3), (4, 3)),
        ((1, 3), (4, 1)),
        ((4, 1), (4, 3)),
        ((4, 1), (1, 3)),
    ],
)
@pytest.mark.parametrize(
    'mat_shape1, mat_shape2',
    [
        ((1, 3), (3, 1)),
        ((2, 1), (1, 2)),
        ((2, 3), (3, 2)),
    ],
)
def test_rule_mul(shape1, shape2, mat_shape1, mat_shape2):
    shapein = broadcast_shapes(shape1 + mat_shape2[1:], shape2 + mat_shape2[1:])
    data1 = np.arange(product(shape1 + mat_shape1)).reshape(shape1 + mat_shape1)
    data2 = np.arange(product(shape2 + mat_shape2)).reshape(shape2 + mat_shape2)
    op1 = DenseBlockDiagonalOperator(data1)
    op2 = DenseBlockDiagonalOperator(data2)
    comp1 = op1 @ op2
    assert isinstance(comp1, DenseBlockDiagonalOperator)
    with rule_manager(none=True):
        comp2 = op1 @ op2
    assert_same(comp1.todense(shapein), comp2.todense(shapein))
