import numpy as np
from pyoperators.utils.testing import assert_eq
from pyoperators.utils.ufuncs import masking
from .common import DTYPES

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
