import itertools
import numpy as np

from pyoperators import Operator
from pyoperators.utils import (
    cast,
    find,
    first_is_not,
    inspect_special_values,
    interruptible,
    isscalar,
    least_greater_multiple,
    product,
    strenum,
    strplural,
    strshape,
    uninterruptible,
)
from pyoperators.utils.testing import assert_eq, assert_raises

dtypes = [
    np.dtype(t)
    for t in (
        np.bool8,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
        np.float128,
        np.complex64,
        np.complex128,
        np.complex256,
    )
]


def assert_dtype(a, d):
    if a is None:
        return
    assert_eq(a.dtype, d)


def assert_is_scalar(o):
    assert isscalar(o)


def assert_is_not_scalar(o):
    assert not isscalar(o)


def test_cast():
    dtypes = (float, complex, None)

    def func(d1, d2):
        a1 = None if d1 is None else np.array(1, dtype=d1)
        a2 = None if d2 is None else np.array(1, dtype=d2)
        if d1 is None and d2 is None:
            assert_raises(ValueError, cast, [a1, a2])
            return
        expected = d1 if d2 is None else d2 if d1 is None else np.promote_types(d1, d2)
        a1_, a2_ = cast([a1, a2])
        assert_dtype(a1_, expected)
        assert_dtype(a2_, expected)

    for d1, d2 in itertools.product(dtypes, repeat=2):
        yield func, d1, d2


def test_find1():
    assert find([1, 2, 3], lambda x: x > 1.5) == 2


def test_find2():
    assert_raises(ValueError, find, [1, 2, 3], lambda x: x > 3)


def test_first_is_not():
    assert first_is_not([1, 2], 1) == 2
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
        for x in (
            (1, 1.1, 0, -1, -1),
            (-1, -1),
            (0, 0),
            (1, 1),
            (2, 2),
            (2, 1),
            np.random.random_integers(-1, 1, size=10),
        ):
            x = np.asarray(x).astype(d)
            yield func, x


def test_interruptible():
    import signal

    def func_interruptible():
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def func_uninterruptible():
        assert signal.getsignal(signal.SIGINT) is not signal.default_int_handler

    with interruptible():
        func_interruptible()
        with uninterruptible():
            func_uninterruptible()
            with uninterruptible():
                func_uninterruptible()
                with interruptible():
                    func_interruptible()
                    with interruptible():
                        func_interruptible()
                    func_interruptible()
                func_uninterruptible()
            func_uninterruptible()
        func_interruptible()


def test_is_scalar():
    for o in (object, True, 1, 1.0, np.array(1), np.int8, slice, Operator()):
        yield assert_is_scalar, o


def test_is_not_scalar():
    for o in ([], (), np.ones(1), np.ones(2)):
        yield assert_is_not_scalar, o


def test_least_greater_multiple():
    def func(lgm, expected):
        assert_eq(lgm, expected)

    a, b, c = np.ogrid[[slice(4, 11) for m in range(3)]]
    expected = 2**a * 3**b * 5**c
    yield func, least_greater_multiple(expected, [2, 3, 5]), expected
    yield func, least_greater_multiple(expected - 1, [2, 3, 5]), expected


def test_product():
    def func(o):
        assert o == 1

    for o in ([], (), (1,), [1], [2, 0.5], (2, 0.5), np.array(1), np.array([2, 0.5])):
        yield func, product(o)


def test_strenum():
    assert_eq(strenum(['blue', 'red', 'yellow'], 'or'), "'blue', 'red' or 'yellow'")


def test_strplural():
    def func(n, nonumber, s, expected):
        assert_eq(strplural(n, 'cat', nonumber=nonumber, s=s), expected)

    for n, nonumber, s, expected in zip(
        4 * (0, 1, 2),
        3 * (True,) + 3 * (False,) + 3 * (True,) + 3 * (False,),
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
        yield func, n, nonumber, s, expected


def test_strshape():
    for shape, expected in zip(((1,), (2, 3)), ('1', '(2,3)')):
        yield assert_eq, strshape(shape), expected
