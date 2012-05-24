from __future__ import division

import numpy as np

from pyoperators import (
    IdentityOperator,
    ZeroOperator,
    DiagonalOperator,
    DenseOperator,
    MaskOperator,
    PackOperator,
    UnpackOperator,
)
from pyoperators.utils.testing import assert_eq


def test_denseoperator():
    def func(m, d, v):
        expected = np.dot(m, v)
        assert_eq(d(v), expected)
        if d.flags.square:
            w = v.copy()
            d(w, w)
            assert_eq(w, expected)

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
    for v in np.array([1 + 0j, 0, 0]), np.array([0j, 1, 0]), np.array([0j, 0, 1]):
        yield func, m.T, d.T, v
        yield func, m.T.conj(), d.H, v


def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1, 2, 3, 4, 5]), [1, 4])
    assert np.allclose(p.T([1, 4]), [1, 0, 0, 4, 0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1, 4]), [1, 0, 0, 4, 0])
    assert np.allclose(u.T([1, 2, 3, 4, 5]), [1, 4])
