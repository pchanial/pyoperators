from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_warns
from pyoperators import (
    BlockDiagonalOperator, HomothetyOperator, PyOperatorsWarning)
from pyoperators.linear import DenseOperator, DenseBlockDiagonalOperator
from pyoperators.rules import rule_manager
from pyoperators.utils import broadcast_shapes, product, reshape_broadcast
from pyoperators.utils.testing import (assert_is_instance, assert_is_type,
                                       assert_same)


def test_dense1():
    def func(m, d, v):
        expected = np.dot(m, v)
        assert_same(d(v), expected)
        if d.flags.square:
            w = v.copy()
            d(w, w)
            assert_same(w, expected)

    m = np.array([[1, 1j], [2, 2]])
    d = DenseOperator(m)
    for v in np.array([1+0j, 0]), np.array([0+0j, 1]):
        yield func, m, d, v
        yield func, m.T, d.T, v
        yield func, m.T.conj(), d.H, v

    m = np.array([[1, 2], [3, 4j], [5, 6]])
    d = DenseOperator(m)
    for v in np.array([1+0j, 0]), np.array([0+0j, 1]):
        yield func, m, d, v
    for v in (np.array([1+0j, 0, 0]), np.array([0j, 1, 0]),
              np.array([0j, 0, 1])):
        yield func, m.T, d.T, v
        yield func, m.T.conj(), d.H, v


def test_dense2():
    shapeins = ((2,), (3, 2), (3, 1, 2))
    shapeouts = ((3,), (2, 3), (2, 1, 3))
    extrainputs = ((), (5,), (3, 4))

    def func(shapein, shapeout, extrainput):
        datashape = shapeout + shapein
        inputshape = extrainput + shapein
        d = np.arange(product(datashape)).reshape(datashape)
        b = DenseOperator(
            d, naxesin=len(shapein), naxesout=len(shapeout),
            shapein=inputshape)
        bdense = b.todense()
        n = product(extrainput)
        d_ = d.reshape((product(shapeout), product(shapein)))
        expected = BlockDiagonalOperator(n * [d_], axisin=0).todense()
        assert_equal(bdense, expected)
    for shapein in shapeins:
        for shapeout in shapeouts:
            for extrainput in extrainputs:
                yield func, shapein, shapeout, extrainput


def test_dense_error():
    shapes = ((2,), (3, 2))
    data = (np.arange(product(s)).reshape(s) for s in shapes)

    def func(d):
        b = DenseOperator(d)
        assert_raises(ValueError, b, np.ones(3))
    for d in data:
        yield func, d


def test_dense_rule_homothety():
    m = np.array([[1, 2], [3, 4], [5, 6]])
    d = HomothetyOperator(2) * DenseOperator(m)
    assert_is_type(d, DenseOperator)
    assert_same(d.data, m * 2)
    d = HomothetyOperator(2j) * DenseOperator(m)
    assert_is_type(d, DenseOperator)
    assert_same(d.data, m * 2j)
    assert_equal(d.dtype, complex)


def test_block_diagonal():
    shapeins = (2,), (3, 2)
    shapeouts = (3,), (2, 3)
    extradatas = (4,), (2, 1), (2, 4)
    extrainputs = (), (4,), (2, 4), (2, 1), (3, 1, 4)

    def func(shapein, shapeout, extradata, extrainput):
        datashape = extradata + shapeout + shapein
        d = np.arange(product(datashape)).reshape(datashape)
        b = DenseBlockDiagonalOperator(
            d, naxesin=len(shapein), naxesout=len(shapeout))
        new_shape = broadcast_shapes(extradata, extrainput)
        bdense = b.todense(shapein=new_shape + shapein)
        d_ = reshape_broadcast(d, new_shape + shapeout + shapein)
        d_ = d_.reshape(-1, product(shapeout), product(shapein))
        expected = BlockDiagonalOperator([_ for _ in d_], axisin=0).todense(
            shapein=product(new_shape + shapein))
        assert_same(bdense, expected)
        bTdense = b.T.todense(shapein=new_shape + shapeout)
        assert_same(bTdense, expected.T)
    for shapein in shapeins:
        for shapeout in shapeouts:
            for extradata in extradatas:
                for extrainput in extrainputs:
                    yield func, shapein, shapeout, extradata, extrainput


def test_morphing():
    def func1(cls):
        d = cls(3.)
        assert_is_type(d, HomothetyOperator)
    for cls in DenseBlockDiagonalOperator, DenseOperator:
        yield func1, cls

    def func2(shape):
        d = DenseBlockDiagonalOperator(np.ones(shape))
        assert_is_type(d, DenseOperator)
    for shape in (3,), (1, 3), (2, 3):
        yield func2, shape


def test_warning():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    a = a.swapaxes(0, 1)
    assert_warns(PyOperatorsWarning, DenseOperator, a, naxesin=2)


def test_rule_mul():
    shapes1 = (), (3,), (3,), (3,), (1,), (1, 3), (1, 3), (4, 1), (4, 1)
    shapes2 = (3,), (), (3,), (1,), (3,), (4, 3), (4, 1), (4, 3), (1, 3)
    mat_shapes1 = (1, 3), (2, 1), (2, 3)
    mat_shapes2 = (3, 1), (1, 2), (3, 2)

    def func(s1, s2, sm1, sm2):
        shapein = broadcast_shapes(s1 + sm2[1:], s2 + sm2[1:])
        data1 = np.arange(product(s1 + sm1)).reshape(s1 + sm1)
        data2 = np.arange(product(s2 + sm2)).reshape(s2 + sm2)
        op1 = DenseBlockDiagonalOperator(data1)
        op2 = DenseBlockDiagonalOperator(data2)
        comp1 = op1 * op2
        assert_is_instance(comp1, DenseBlockDiagonalOperator)
        with rule_manager(none=True):
            comp2 = op1 * op2
        assert_equal(comp1.todense(shapein), comp2.todense(shapein))
    for s1, s2 in zip(shapes1, shapes2):
        for sm1, sm2 in zip(mat_shapes1, mat_shapes2):
            yield func, s1, s2, sm1, sm2
