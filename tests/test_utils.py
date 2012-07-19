import numpy as np

from pyoperators import Operator
from pyoperators.utils import (find, first_is_not, inspect_special_values,
                               isscalar, least_greater_multiple, product,
                               strenum, strplural, strshape)
from pyoperators.utils.testing import assert_eq, assert_is_none

dtypes = [np.dtype(t) for t in (np.bool8, np.uint8, np.int8, np.uint16,
          np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32,
          np.float64, np.float128, np.complex64, np.complex128, np.complex256)]

def assert_is_scalar(o):
    assert isscalar(o)

def assert_is_not_scalar(o):
    assert not isscalar(o)

def test_find1():
    assert find([1,2,3], lambda x: x > 1.5) == 2

def test_find2():
    assert_is_none(find([1,2,3], lambda x: x > 3))

def test_first_is_not():
    assert first_is_not([1,2], 1) == 2
    assert first_is_not([None, None, {}], None) == {}
    assert first_is_not([], None) is None
    assert first_is_not([None, None], None) is None

def test_inspect_special_values():
    def ref(x):
        nones = nzeros = nmones = nothers = 0
        for value in x.flat:
            if value == 0:
                nzeros += 1
            elif value == 1:
                nones += 1
            elif value == -1:
                nmones += 1
            else:
                nothers += 1
        if nothers > 0:
            nmones = nzeros = nones = 0
        return nmones, nzeros, nones, nothers > 0, np.all(x == x.flat[0])

    def func(x):
        assert_eq(inspect_special_values(x), ref(x))
    for d in dtypes:
        for x in ((1, 1.1, 0, -1, -1), (-1, -1), (0, 0), (1,1), (2, 2), (2,1),
                  np.random.random_integers(-1,1,size=10)):
            x = np.asarray(x).astype(d)
            yield func, x

def test_is_scalar():
    for o in (object, True, 1, 1., np.array(1), np.int8, slice, Operator()):
        yield assert_is_scalar, o

def test_is_not_scalar():
    for o in ([],(), np.ones(1), np.ones(2)):
        yield assert_is_not_scalar, o

def test_least_greater_multiple():
    def func(lgm, expected):
        assert_eq(lgm, expected)
    a, b, c = np.ogrid[[slice(4, 11) for m in range(3)]]
    expected = 2**a * 3**b * 5**c
    yield func, least_greater_multiple(expected, [2,3,5]), expected
    yield func, least_greater_multiple(expected-1, [2,3,5]), expected

def test_product():
    def func(o):
        assert o == 1
    for o in ([],(),(1,),[1],[2,0.5],(2,0.5),np.array(1),np.array([2,0.5])):
        yield func, product(o)

def test_strenum():
    assert_eq(strenum(['blue', 'red', 'yellow'], 'or'),
              "'blue', 'red' or 'yellow'")

def test_strplural():
    def func(n, prepend, s, expected):
        assert_eq(strplural('cat', n, prepend=prepend, s=s), expected)
    for n, prepend, s, expected in zip(
        4*(0,1,2),
        3*(False,) + 3*(True,) + 3*(False,) + 3*(True,),
        6*('',) + 6*(':',),
        ('cat', 'cat', 'cats', 'no cat', '1 cat', '2 cats',
         'cat', 'cat:', 'cats:', 'no cat', '1 cat:', '2 cats:')):
        yield func, n, prepend, s, expected

def test_strshape():
    for shape, expected in zip(((1,), (2,3)), ('1', '(2,3)')):
        yield assert_eq, strshape(shape), expected
