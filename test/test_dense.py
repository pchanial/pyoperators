from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_raises
from pyoperators import (
    BlockColumnOperator,
    BlockDiagonalOperator,
    DenseOperator,
    HomothetyOperator,
)
from pyoperators.linear import DenseBlockColumnOperator, DenseBlockDiagonalOperator
from pyoperators.utils import product
from pyoperators.utils.testing import assert_is_type, assert_same


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
    for v in np.array([1 + 0j, 0]), np.array([0 + 0j, 1]):
        yield func, m, d, v
        yield func, m.T, d.T, v
        yield func, m.T.conj(), d.H, v

    m = np.array([[1, 2], [3, 4j], [5, 6]])
    d = DenseOperator(m)
    for v in np.array([1 + 0j, 0]), np.array([0 + 0j, 1]):
        yield func, m, d, v
    for v in (np.array([1 + 0j, 0, 0]), np.array([0j, 1, 0]), np.array([0j, 0, 1])):
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
            d, naxesin=len(shapein), naxesout=len(shapeout), shapein=inputshape
        )
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
    shapeins = ((2,), (3, 2))
    shapeouts = ((3,), (2, 3))
    extradatas = ((4,), (2, 4))
    extrainputs = ((), (5,), (3, 4))

    def func(shapein, shapeout, extradata, extrainput):
        cls = DenseBlockDiagonalOperator
        datashape = extradata + shapeout + shapein
        if cls is DenseBlockDiagonalOperator:
            inputshape = extrainput + extradata + shapein
        else:
            inputshape = extrainput + shapein
        d = np.arange(product(datashape)).reshape(datashape)
        b = DenseBlockDiagonalOperator(
            d, naxesin=len(shapein), naxesout=len(shapeout), shapein=inputshape
        )
        bdense = b.todense()
        n = product(shapein)
        n = product(extrainput)
        d_ = d.reshape((product(extradata), product(shapeout), product(shapein)))
        expected = BlockDiagonalOperator(
            n * [BlockDiagonalOperator([_ for _ in d_], axisin=0)], axisin=0
        ).todense()
        assert_same(bdense, expected)

    for shapein in shapeins:
        for shapeout in shapeouts:
            for extradata in extradatas:
                for extrainput in extrainputs:
                    yield func, shapein, shapeout, extradata, extrainput


def test_block_column():
    shapeins = ((2,), (3, 2), (3, 1, 2))
    shapeouts = ((3,), (2, 3), (2, 1, 3))
    extradatas = ((), (4,), (2, 4))
    extrainputs = ((), (5,), (3, 4))

    def func(shapein, shapeout, extradata, extrainput, broadcast):
        datashape = extradata + shapeout + shapein
        inputshape = extrainput + shapein
        d = np.arange(product(datashape)).reshape(datashape)
        b = DenseBlockColumnOperator(
            d,
            naxesin=len(shapein),
            naxesout=len(shapeout),
            broadcast=broadcast,
            shapein=inputshape,
        )
        if broadcast == 'outer':
            assert b.shapeout == extrainput + extradata + shapeout
        else:
            assert b.shapeout == extradata + extrainput + shapeout
        bdense = b.todense()
        n = product(shapein)
        if broadcast == 'outer':
            expected = BlockDiagonalOperator(
                product(extrainput) * [DenseOperator(d.reshape((-1, n)), shapein=n)],
                axisin=0,
            ).todense()
        else:
            expected = BlockColumnOperator(
                [
                    BlockDiagonalOperator(
                        product(extrainput)
                        * [DenseOperator(_.reshape((-1, n)), shapein=n)],
                        axisin=0,
                    )
                    for _ in d.reshape((product(extradata), -1, n))
                ],
                axisout=0,
            ).todense()

        assert_same(bdense, expected)
        bTdense = b.T.todense()
        assert_same(bTdense, bdense.T)

        b2 = DenseBlockColumnOperator(
            d, naxesin=len(shapein), naxesout=len(shapeout), broadcast=broadcast
        )
        assert_same(b2.todense(shapein=inputshape), bdense)
        assert_same(b2.T.todense(shapein=b.T.shapein), bTdense)

    for shapein in shapeins:
        for shapeout in shapeouts:
            for extradata in extradatas:
                for extrainput in extrainputs:
                    for broadcast in ('inner', 'outer'):
                        yield (
                            func,
                            shapein,
                            shapeout,
                            extradata,
                            extrainput,
                            broadcast,
                        )
