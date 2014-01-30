import itertools
import numpy as np
from pyoperators.utils import pi
from pyoperators.utils.testing import assert_eq, assert_same
from pyoperators.utils.ufuncs import abs2, masking, multiply_conjugate
from .common import DTYPES, COMPLEX_DTYPES


def test_abs2():
    x = np.array([pi(np.float128) + 1j, pi(np.float128)*1j, 3])

    def func(d):
        x_ = np.array(x if d.kind == 'c' else x.real, dtype=d)
        actual = abs2(x_)
        expected = np.abs(x_**2)
        assert_same(actual, expected)
        abs2(x_, actual)
        assert_same(actual, expected)
    for dtype in DTYPES:
        yield func, dtype


def test_masking():
    def func(a, mask):
        actual = masking(a, mask)
        expected = a.copy()
        expected[mask] = 0
        assert_eq(actual, expected)
        masking(a, mask, a)
        assert_eq(a, expected)
    for dtype in DTYPES:
        a = np.arange(4, dtype=dtype)
        mask = np.array([True, False, False, True], dtype=bool)
        yield func, a, mask


def test_multiply_conjugate():
    def func(x1, x2, cdtype):
        result = multiply_conjugate(x1, x2)
        expected = x1 * x2.conjugate()
        assert_eq(result, expected)
        result[...] = 0
        multiply_conjugate(x1, x2, result)
        assert_eq(result, expected)
    for dtype, cdtype in itertools.product(DTYPES, COMPLEX_DTYPES):
        x1 = np.array([1+1j, 1+1j, 3+1j])
        if dtype.kind == 'c':
            x1 = x1.astype(dtype)
        else:
            x1 = x1.real.astype(dtype)
        x2 = np.array(1-1j, dtype=cdtype)
        yield func, x1, x2, cdtype
