from __future__ import division

import numpy as np
import operator
import scipy.sparse as sp
from numpy.testing import assert_raises
from pyoperators import SparseOperator
from pyoperators.utils.testing import assert_same

A = np.array([[1, 0, 2, 0],
              [0, 0, 3, 0],
              [4, 5, 6, 0],
              [1, 0, 0, 1]])
vecs = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
formats = 'bsr,csr,csc,coo,dia,dok'.split(',')


def test_matvec():
    def func(format):
        cls = getattr(sp, format + '_matrix')
        so = SparseOperator(cls(A))
        for vec in vecs:
            assert_same(so(vec), np.dot(A, vec))
            assert_same(so.T(vec), np.dot(A.T, vec))
        assert_same(so.todense(), A)
        assert_same(so.T.todense(), A.T)
    for format in formats:
        yield func, format


def test_shapes():
    def func(format):
        cls = getattr(sp, format + '_matrix')
        shapein = (2, 2)
        shapeout = (1, 4, 1)
        so = SparseOperator(cls(A), shapein=shapein, shapeout=shapeout)
        for vec in vecs:
            assert_same(so(np.reshape(vec, shapein)),
                        np.dot(A, vec).reshape(shapeout))
            assert_same(so.T(np.reshape(vec, shapeout)),
                        np.dot(A.T, vec).reshape(shapein))
    for format in formats:
        yield func, format


def test_update_output():
    def func(format):
        cls = getattr(sp, format + '_matrix')
        so = SparseOperator(cls(A))
        out = np.zeros(4, dtype=int)
        outT = np.zeros(4, dtype=int)
        for vec in vecs:
            so(vec, out, operation=operator.iadd)
            so.T(vec, outT, operation=operator.iadd)
        assert_same(out, np.sum(A, axis=1))
        assert_same(outT, np.sum(A, axis=0))
    for format in formats:
        yield func, format


def test_error1():
    values = (sp.lil_matrix(A), np.zeros((10, 4)),
              np.matrix(np.zeros((10, 4))), 3)

    def func(v):
        assert_raises(TypeError, SparseOperator, v)
    for v in values:
        yield func, v


def test_error2():
    def func(format):
        cls = getattr(sp, format + '_matrix')
        sm = cls(A)
        shapein = (2, 3)
        shapeout = (1, 4, 2)
        assert_raises(ValueError, SparseOperator, sm, shapein=shapein)
        assert_raises(ValueError, SparseOperator, sm, shapeout=shapeout)
    for format in formats:
        yield func, format


def test_error3():
    def func(format):
        cls = getattr(sp, format + '_matrix')
        sm = cls(A)
        so = SparseOperator(sm)
        out = np.zeros(4, dtype=int)
        assert_raises(ValueError, so, vecs[0], out, operation=operator.imul)
    for format in formats:
        yield func, format
