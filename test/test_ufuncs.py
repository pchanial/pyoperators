import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators.utils import pi
from pyoperators.utils.testing import assert_same
from pyoperators.utils.ufuncs import abs2, masking, multiply_conjugate

from .common import BIGGEST_FLOAT_TYPE, COMPLEX_DTYPES, DTYPES


@pytest.mark.parametrize('dtype', DTYPES)
def test_abs2(dtype):
    x = np.array([pi(BIGGEST_FLOAT_TYPE) + 1j, pi(BIGGEST_FLOAT_TYPE) * 1j, 3])
    x_ = np.array(x if dtype.kind == 'c' else x.real, dtype=dtype)
    actual = abs2(x_)
    expected = np.abs(x_**2)
    assert_same(actual, expected)
    abs2(x_, actual)
    assert_same(actual, expected)


@pytest.mark.parametrize('dtype', DTYPES)
def test_masking(dtype):
    a = np.arange(4, dtype=dtype)
    mask = np.array([True, False, False, True], dtype=bool)
    actual = masking(a, mask)
    expected = a.copy()
    expected[mask] = 0
    assert_equal(actual, expected)
    masking(a, mask, a)
    assert_equal(a, expected)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('cdtype', COMPLEX_DTYPES)
def test_multiply_conjugate(dtype, cdtype):
    x1 = np.array([1 + 1j, 1 + 1j, 3 + 1j])
    if dtype.kind == 'c':
        x1 = x1.astype(dtype)
    else:
        x1 = x1.real.astype(dtype)
    x2 = np.array(1 - 1j, dtype=cdtype)
    result = multiply_conjugate(x1, x2)
    expected = x1 * x2.conjugate()
    assert_equal(result, expected)
    result[...] = 0
    multiply_conjugate(x1, x2, result)
    assert_equal(result, expected)
