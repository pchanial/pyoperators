import operator

import numpy as np
import pytest
import scipy.sparse as sp

from pyoperators import SparseOperator
from pyoperators.utils.testing import assert_same

A = np.array([[1, 0, 2, 0], [0, 0, 3, 0], [4, 5, 6, 0], [1, 0, 0, 1]])
VECTORS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
FORMATS = 'bsr,csr,csc,coo,dia,dok'.split(',')


@pytest.mark.parametrize('format', FORMATS)
def test_matvec(format):
    cls = getattr(sp, format + '_matrix')
    so = SparseOperator(cls(A))
    for vec in VECTORS:
        assert_same(so(vec), np.dot(A, vec))
        assert_same(so.T(vec), np.dot(A.T, vec))
    assert_same(so.todense(), A)
    assert_same(so.T.todense(), A.T)


@pytest.mark.parametrize('format', FORMATS)
@pytest.mark.parametrize('vector', VECTORS)
def test_shapes(format, vector):
    cls = getattr(sp, format + '_matrix')
    shapein = (2, 2)
    shapeout = (1, 4, 1)
    so = SparseOperator(cls(A), shapein=shapein, shapeout=shapeout)
    assert_same(so(np.reshape(vector, shapein)), np.dot(A, vector).reshape(shapeout))
    assert_same(
        so.T(np.reshape(vector, shapeout)), np.dot(A.T, vector).reshape(shapein)
    )


@pytest.mark.parametrize('format', FORMATS)
def test_update_output(format):
    cls = getattr(sp, format + '_matrix')
    so = SparseOperator(cls(A))
    out = np.zeros(4, dtype=int)
    outT = np.zeros(4, dtype=int)
    for vector in VECTORS:
        so(vector, out, operation=operator.iadd)
        so.T(vector, outT, operation=operator.iadd)
    assert_same(out, np.sum(A, axis=1))
    assert_same(outT, np.sum(A, axis=0))


@pytest.mark.parametrize(
    'value', [sp.lil_matrix(A), np.zeros((10, 4)), np.array(np.zeros((10, 4))), 3]
)
def test_error1(value):
    with pytest.raises(TypeError):
        SparseOperator(value)


@pytest.mark.parametrize('format', FORMATS)
def test_error2(format):
    cls = getattr(sp, format + '_matrix')
    sm = cls(A)
    shapein = (2, 3)
    shapeout = (1, 4, 2)
    with pytest.raises(ValueError):
        SparseOperator(sm, shapein=shapein)
    with pytest.raises(ValueError):
        SparseOperator(sm, shapeout=shapeout)


@pytest.mark.parametrize('format', FORMATS)
def test_error3(format):
    cls = getattr(sp, format + '_matrix')
    sm = cls(A)
    so = SparseOperator(sm)
    out = np.zeros(4, dtype=int)
    with pytest.raises(ValueError):
        so(VECTORS[0], out, operation=operator.imul)
