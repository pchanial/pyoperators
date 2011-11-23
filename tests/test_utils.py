import numpy as np
from nose.tools import eq_

from pyoperators import Operator
from pyoperators.utils import first_is_not, isscalar, strenum, strplural, strshape


def assert_is_scalar(o):
    assert isscalar(o)


def assert_is_not_scalar(o):
    assert not isscalar(o)


def test_first_is_not():
    assert first_is_not([1, 2], 1) == 2
    assert first_is_not([None, None, {}], None) == {}
    assert first_is_not([], None) is None
    assert first_is_not([None, None], None) is None


def test_is_scalar():
    for o in (object, True, 1, 1.0, np.array(1), np.int8, slice, Operator()):
        yield assert_is_scalar, o


def test_is_not_scalar():
    for o in ([], (), np.ones(1), np.ones(2)):
        yield assert_is_not_scalar, o


def test_strenum():
    eq_(strenum(['blue', 'red', 'yellow'], 'or'), "'blue', 'red' or 'yellow'")


def test_strplural():
    def func(n, prepend, s, expected):
        eq_(strplural('cat', n, prepend=prepend, s=s), expected)

    for n, prepend, s, expected in zip(
        4 * (0, 1, 2),
        3 * (False,) + 3 * (True,) + 3 * (False,) + 3 * (True,),
        6 * ('',) + 6 * (':',),
        (
            'cat',
            'cat',
            'cats',
            'no cat',
            '1 cat',
            '2 cats',
            'cat',
            'cat:',
            'cats:',
            'no cat',
            '1 cat:',
            '2 cats:',
        ),
    ):
        yield func, n, prepend, s, expected


def test_strshape():
    for shape, expected in zip(((1,), (2, 3)), ('1', '(2,3)')):
        yield eq_, strshape(shape), expected
