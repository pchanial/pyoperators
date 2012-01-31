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
from pyoperators.utils import assert_eq


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


def test_masking():

    mask = MaskOperator(0)
    assert isinstance(mask, IdentityOperator)
    mask = MaskOperator(0, shapein=(32, 32), dtype=np.float32)
    assert isinstance(mask, IdentityOperator)
    assert mask.shapein == (32, 32)
    assert mask.dtype == np.float32

    mask = MaskOperator(1)
    assert isinstance(mask, ZeroOperator)
    mask = MaskOperator(1, shapein=(32, 32), dtype=np.float32)
    assert isinstance(mask, ZeroOperator)
    assert mask.shapein == (32, 32)
    assert mask.dtype == np.float32

    b = np.array([3.0, 4.0, 1.0, 0.0, 3.0, 2.0])
    c = np.array([3.0, 4.0, 0.0, 0.0, 3.0, 0.0])
    mask = MaskOperator(np.array([0, 0.0, 1.0, 1.0, 0.0, 1], dtype=np.int8))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([1, 1.0, 0.0, 0.0, 1.0, 0]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([False, False, True, True, False, True]))
    assert np.all(mask(b) == c)

    b = np.array([[3.0, 4.0], [1.0, 0.0], [3.0, 2.0]])
    c = np.array([[3.0, 4.0], [0.0, 0.0], [3.0, 0.0]])
    mask = MaskOperator(np.array([[0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype='int8'))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([[1, 1.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([[False, False], [True, True], [False, True]]))
    assert np.all(mask(b) == c)

    b = np.array([[[3, 4.0], [1.0, 0.0]], [[3.0, 2], [-1, 9]]])
    c = np.array([[[3, 4.0], [0.0, 0.0]], [[3.0, 0], [0, 0]]])
    mask = MaskOperator(np.array([[[0, 0.0], [1.0, 1.0]], [[0.0, 1], [1, 1]]], int))
    assert np.all(mask(b) == c)

    mask = DiagonalOperator(np.array([[[1, 1], [0.0, 0]], [[1, 0], [0, 0]]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(
        np.array([[[False, False], [True, True]], [[False, True], [True, True]]])
    )
    assert np.all(mask(b) == c)

    c = mask(b, b)
    assert id(b) == id(c)


def test_masking2():
    m = MaskOperator([True, False, True])
    assert m * m is m


def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1, 2, 3, 4, 5]), [1, 4])
    assert np.allclose(p.T([1, 4]), [1, 0, 0, 4, 0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1, 4]), [1, 0, 0, 4, 0])
    assert np.allclose(u.T([1, 2, 3, 4, 5]), [1, 4])
