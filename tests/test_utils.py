import os
import time

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import Operator
from pyoperators.utils import (
    Timer,
    broadcast_shapes,
    cast,
    complex_dtype,
    deprecated,
    first,
    first_is_not,
    float_dtype,
    float_intrinsic_dtype,
    float_or_complex_dtype,
    ifirst,
    ifirst_is_not,
    ilast,
    ilast_is_not,
    inspect_special_values,
    interruptible,
    isscalarlike,
    last,
    last_is_not,
    least_greater_multiple,
    omp_num_threads,
    one,
    pi,
    pool_threading,
    product,
    reshape_broadcast,
    setting,
    settingerr,
    split,
    strenum,
    strplural,
    strshape,
    uninterruptible,
    zero,
    zip_broadcast,
)
from pyoperators.utils.testing import assert_same
from pyoperators.warnings import PyOperatorsDeprecationWarning

from .common import BIGGEST_FLOAT_TYPE, COMPLEX_DTYPES, DTYPES, FLOAT_DTYPES


def test_deprecated_func():
    @deprecated('blabla')
    def myfunc():
        return 2

    with pytest.warns(PyOperatorsDeprecationWarning):
        val = myfunc()
    assert val == 2


def test_deprecated_func_error():
    with pytest.raises(TypeError):
        deprecated(lambda x: x)


def test_deprecated_class():
    class mynewclass:
        def __init__(self, a, b=None):
            self.a = a
            self.b = b

    @deprecated
    class myclass1(mynewclass):
        pass

    with pytest.warns(PyOperatorsDeprecationWarning):
        c = myclass1(3, b=2)
    assert c.a == 3
    assert c.b == 2

    @deprecated('blabla2')
    class myclass2(mynewclass):
        pass

    with pytest.warns(PyOperatorsDeprecationWarning):
        c = myclass2(3, b=2)
    assert c.a == 3
    assert c.b == 2


@pytest.mark.parametrize(
    'shape, expected',
    [
        (((),), ()),
        (((), ()), ()),
        (((), (), ()), ()),
        (((1,),), (1,)),
        (((), (1,)), (1,)),
        (((), (1,), (1,)), (1,)),
        (((2,),), (2,)),
        (((), (2,)), (2,)),
        (((1,), (2,)), (2,)),
        (((), (2,), (2,)), (2,)),
        (((), (2,), (1,)), (2,)),
        (((2,), (1,), ()), (2,)),
        (((1,), (2, 1)), (2, 1)),
        (((), (2, 1)), (2, 1)),
        (((1,), (2, 1), ()), (2, 1)),
        (((1,), (1, 2)), (1, 2)),
        (((), (1, 2)), (1, 2)),
        (((1,), (1, 2), ()), (1, 2)),
        (((2,), (2, 1)), (2, 2)),
        (((), (2,), (2, 1)), (2, 2)),
        (((2,), (2, 1), ()), (2, 2)),
        (((1, 2), (2, 1)), (2, 2)),
        (((), (1, 2), (2, 1)), (2, 2)),
        (((1, 2), (2, 1), ()), (2, 2)),
        (((1, 1, 4), (1, 3, 1), (2, 1, 1), (), (1, 1, 1)), (2, 3, 4)),
    ],
)
def test_broadcast_shapes(shape, expected):
    assert broadcast_shapes(*shape) == expected


@pytest.mark.parametrize('d1', [float, complex, np.int8])
@pytest.mark.parametrize('d2', [float, complex, np.int8])
def test_cast(d1, d2):
    a1 = np.array(1, dtype=d1)
    a2 = np.array(1, dtype=d2)
    expected = np.promote_types(d1, d2)
    a1_, a2_ = cast([a1, a2])
    assert a1_.dtype == expected
    assert a2_.dtype == expected


def test_cast_invalid():
    with pytest.raises(ValueError):
        cast([None, None])


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (bool, '<c16'),
        (int, '<c16'),
        (np.uint32, '<c16'),
        (np.float16, '<c16'),
        (np.float32, '<c8'),
        (np.float64, '<c16'),
        (np.complex64, '<c8'),
        (np.complex128, '<c16'),
        ('>f2', '<c16'),
        ('>f4', '>c8'),
        ('>f8', '>c16'),
        ('>c8', '>c8'),
        ('>c16', '>c16'),
    ]
    + (
        [
            (np.float128, '<c32'),
            ('>f16', '>c32'),
            (np.complex256, '<c32'),
            ('>c32', '>c32'),
        ]
        if hasattr(np, 'float128')
        else []
    ),
)
def test_complex_dtype(dtype, expected):
    actual = complex_dtype(dtype)
    assert actual.str == expected


def test_complex_dtype_invalid():
    with pytest.raises(TypeError):
        complex_dtype(str)


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (bool, '<f8'),
        (int, '<f8'),
        (np.uint32, '<f8'),
        (np.float16, '<f2'),
        (np.float32, '<f4'),
        (np.float64, '<f8'),
        (np.complex64, '<f4'),
        (np.complex128, '<f8'),
        ('>f2', '>f2'),
        ('>f4', '>f4'),
        ('>f8', '>f8'),
        ('>c8', '>f4'),
        ('>c16', '>f8'),
    ]
    + (
        [
            (np.float128, '<f16'),
            ('>f16', '>f16'),
            (np.complex256, '<f16'),
            ('>c32', '>f16'),
        ]
        if hasattr(np, 'float128')
        else []
    ),
)
def test_float_dtype(dtype, expected):
    actual = float_dtype(dtype)
    assert actual == expected


def test_float_dtype_invalid():
    with pytest.raises(TypeError):
        float_dtype(str)


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (bool, '<f8'),
        (int, '<f8'),
        (np.uint32, '<f8'),
        (np.float16, '<f4'),
        (np.float32, '<f4'),
        (np.float64, '<f8'),
        (np.complex64, '<f4'),
        (np.complex128, '<f8'),
        ('>f2', '<f4'),
        ('>f4', '<f4'),
        ('>f8', '<f8'),
        ('>c8', '<f4'),
        ('>c16', '<f8'),
    ]
    + (
        [
            (np.float128, '<f8'),
            ('>f16', '<f8'),
            (np.complex256, '<f8'),
            ('>c32', '<f8'),
        ]
        if hasattr(np, 'float128')
        else []
    ),
)
def test_float_intrinsic_dtype(dtype, expected):
    actual = float_intrinsic_dtype(dtype)
    assert actual.str == expected


def test_float_intrinsic_dtype_invalid():
    with pytest.raises(TypeError):
        float_intrinsic_dtype(str)


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (bool, '<f8'),
        (int, '<f8'),
        (np.uint32, '<f8'),
        (np.float16, '<f2'),
        (np.float32, '<f4'),
        (np.float64, '<f8'),
        (np.complex64, '<c8'),
        (np.complex128, '<c16'),
        ('>f2', '>f2'),
        ('>f4', '>f4'),
        ('>f8', '>f8'),
        ('>c8', '>c8'),
        ('>c16', '>c16'),
    ]
    + (
        [
            (np.float128, '<f16'),
            ('>f16', '>f16'),
            (np.complex256, '<c32'),
            ('>c32', '>c32'),
        ]
        if hasattr(np, 'float128')
        else []
    ),
)
def test_float_or_complex_dtype(dtype, expected):
    actual = float_or_complex_dtype(dtype)
    assert actual == expected


def test_float_or_complex_dtype_invalid():
    with pytest.raises(TypeError):
        float_or_complex_dtype(str)


def test_first():
    assert first([1, 2, 3], lambda x: x > 1.5) == 2


def test_last():
    assert last([1, 2, 3], lambda x: x > 1.5) == 3


def test_ifirst():
    assert ifirst([1, 2, 3], lambda x: x > 1.5) == 1
    assert ifirst([1, 2, 2, 3], 2.0) == 1
    assert ifirst([1, None, None, 3], None) == 1


def test_ilast():
    assert ilast([1, 2, 3], lambda x: x > 1.5) == 2
    assert ilast([1, 2, 2, 3], 2.0) == 2
    assert ilast([1, None, None, 3], None) == 2


@pytest.mark.parametrize(
    'iterable, value, expected',
    [
        ([0, 2, 1, 0], 0, 2),
        ([None, None, {}], None, {}),
        ([None, None, ()], None, ()),
        ([None, None, []], None, []),
        ([None, None, set()], None, set()),
        ([None, None], 1, None),
        ([2, 2, 2], 2, 2),
        ([], None, None),
        ([None, None], None, None),
    ],
)
def test_first_is_not(iterable, value, expected):
    assert first_is_not(iterable, value) == expected


@pytest.mark.parametrize(
    'iterable, value, expected',
    [
        ([0, 2, 1, 0], 0, 1),
        ([None, None, {}], None, 2),
        ([None, None, ()], None, 2),
        ([None, None, []], None, 2),
        ([None, None, set()], None, 2),
        ([None, None], 1, 0),
    ],
)
def test_ifirst_is_not(iterable, value, expected):
    assert ifirst_is_not(iterable, value) == expected


@pytest.mark.parametrize(
    'iterable, value, expected',
    [
        ([0, 2, 1, 0], 0, 1),
        ([{}, None, None], None, {}),
        ([(), None, None], None, ()),
        ([[], None, None], None, []),
        ([set(), None, None], None, set()),
        ([None, None], 1, None),
        ([2, 2, 2], 2, 2),
        ([], None, None),
        ([None, None], None, None),
    ],
)
def test_last_is_not(iterable, value, expected):
    assert last_is_not(iterable, value) == expected


@pytest.mark.parametrize(
    'iterable, value, expected',
    [
        ([0, 2, 1, 0], 0, 2),
        ([{}, None, None], None, 0),
        ([(), None, None], None, 0),
        ([[], None, None], None, 0),
        ([set(), None, None], None, 0),
        ([None, None], 1, 1),
    ],
)
def test_ilast_is_not(iterable, value, expected):
    assert ilast_is_not(iterable, value) == expected


@pytest.mark.parametrize('func', [first, last, ifirst, ilast])
def test_first_or_last_error_not_found_func(func):
    with pytest.raises(ValueError):
        func([1, 2, 3], lambda x: x > 3)


@pytest.mark.parametrize('func', [ifirst, ilast])
def test_first_or_last_error_not_found_value(func):
    with pytest.raises(ValueError):
        func([1, 2, 3], 4)


@pytest.mark.parametrize('func', [ifirst_is_not, ilast_is_not])
@pytest.mark.parametrize(
    'iterable, value',
    [
        ([2, 2, 2], 2),
        ([], None),
        ([None, None], None),
    ],
)
def test_first_or_last_error_found(func, iterable, value):
    with pytest.raises(ValueError):
        func(iterable, value)


@pytest.mark.parametrize('dtype', [bool] + DTYPES)
@pytest.mark.parametrize(
    'value',
    [
        (1, 1.1, 0, -1, -1),
        (-1, -1),
        (0, 0),
        (1, 1),
        (2, 2),
        (2, 1),
        np.random.random_integers(-1, 1, size=10),
    ],
)
def test_inspect_special_values(dtype, value):
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

    value = np.asarray(value).astype(dtype)
    assert inspect_special_values(value) == ref(value)


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
    func_interruptible()


@pytest.mark.parametrize('x', [True, 1, 1.0, 'lkj', 'jj', np.array(1)])
def test_is_scalar(x):
    assert isscalarlike(x)


@pytest.mark.parametrize(
    'x',
    [
        [],
        (),
        np.ones((0, 1)),
        np.ones(1),
        np.ones(2),
        object,
        np.int8,
        slice,
        Operator(),
    ],
)
def test_is_not_scalar(x):
    assert not isscalarlike(x)


def test_least_greater_multiple():
    a, b, c = np.ogrid[[slice(4, 11) for m in range(3)]]
    expected = 2**a * 3**b * 5**c
    assert_equal(least_greater_multiple(expected, [2, 3, 5]), expected)
    assert_equal(least_greater_multiple(expected - 1, [2, 3, 5]), expected)


@pytest.mark.parametrize(
    'func, expected',
    [
        (one, 1),
        (pi, 4 * np.arctan(np.array(1, BIGGEST_FLOAT_TYPE))),
        (zero, 0),
    ],
)
@pytest.mark.parametrize('dtype', FLOAT_DTYPES + COMPLEX_DTYPES)
def test_one_pi_zero(func, expected, dtype):
    actual = func(dtype)
    assert actual.dtype == dtype
    assert_same(actual, np.array(expected, dtype=dtype))


@pytest.mark.parametrize('nthread', [None, 1, 3])
def test_pool_threading(mocker, nthread):
    try:
        import mkl

        mkl_nthreads = mkl.get_max_threads()
    except ImportError:
        mkl = None

    counter = 0

    def func_thread(i):
        nonlocal counter
        counter += 1

    environ = mocker.patch.dict(os.environ)
    if nthread is None:
        environ.pop('OMP_NUM_THREADS', None)
    else:
        environ['OMP_NUM_THREADS'] = str(nthread)

    omp_num_threads_env = os.getenv('OMP_NUM_THREADS')
    expected_nthread = omp_num_threads()

    with pool_threading() as pool:
        assert int(os.environ['OMP_NUM_THREADS']) == 1
        if mkl is not None:
            assert mkl.get_max_threads() == 1
        pool.map(func_thread, range(pool._processes))

    assert counter == expected_nthread

    assert os.getenv('OMP_NUM_THREADS') == omp_num_threads_env
    if mkl:
        assert mkl.get_max_threads() == mkl_nthreads


@pytest.mark.parametrize(
    'value', [[], (), (1,), [1], [2, 0.5], (2, 0.5), np.array(1), np.array([2, 0.5])]
)
def test_product(value):
    assert product(value) == 1


@pytest.mark.parametrize(
    'shape, new_shape',
    [
        ((4, 5), (4, 5)),
        ((4, 5), (1, 4, 5)),
        ((4, 5), (2, 4, 5)),
        ((4, 5), (2, 3, 4, 5)),
        ((1, 4, 5), (1, 4, 5)),
        ((1, 4, 5), (2, 4, 5)),
        ((1, 4, 5), (1, 2, 4, 5)),
        ((1, 4, 5), (2, 2, 4, 5)),
        ((1, 4, 5), (2, 3, 2, 4, 5)),
        ((4, 5, 1), (4, 5, 1)),
        ((4, 5, 1), (4, 5, 2)),
        ((4, 5, 1), (1, 4, 5, 2)),
        ((4, 5, 1), (2, 4, 5, 2)),
        ((4, 5, 1), (2, 3, 4, 5, 2)),
    ],
)
def test_reshape_broadcast(shape, new_shape):
    data = np.arange(20)
    data_ = data.reshape(shape)
    expected = np.empty(new_shape)
    expected[...] = data_
    actual = reshape_broadcast(data_, new_shape)
    assert_equal(actual, expected)


def test_setting():
    class Obj:
        pass

    obj = Obj()
    obj.myattr = 'old'
    with setting(obj, 'myattr', 'mid'):
        assert obj.myattr == 'mid'
        with setting(obj, 'myattr', 'new'):
            assert obj.myattr == 'new'
        assert obj.myattr == 'mid'
    assert obj.myattr == 'old'

    with setting(obj, 'otherattr', 'mid'):
        assert obj.otherattr == 'mid'
        with setting(obj, 'otherattr', 'new'):
            assert obj.otherattr == 'new'
            with setting(obj, 'anotherattr', 'value'):
                assert obj.anotherattr == 'value'
            assert not hasattr(obj, 'anotherattr')
        assert obj.otherattr == 'mid'
    assert not hasattr(obj, 'otherattr')


def test_settingerr():
    ref1 = np.seterr()
    ref2 = {
        'divide': 'ignore',
        'invalid': 'ignore',
        'over': 'ignore',
        'under': 'ignore',
    }
    ref3 = {'divide': 'raise', 'invalid': 'ignore', 'over': 'warn', 'under': 'ignore'}

    with settingerr(all='ignore'):
        assert np.seterr() == ref2
        with settingerr(divide='raise', over='warn'):
            assert np.seterr() == ref3
        assert np.seterr() == ref2
    assert np.seterr() == ref1


@pytest.mark.parametrize('n', range(4))
@pytest.mark.parametrize('m', range(1, 6))
def test_split(n, m):
    slices = list(split(n, m))
    assert len(slices) == m
    x = np.zeros(n, int)
    for s in slices:
        x[s] += 1
    assert_same(x, 1, broadcasting=True)
    assert [split(n, m, i) for i in range(m)] == slices


def test_strenum():
    assert strenum(['blue', 'red', 'yellow'], 'or') == "'blue', 'red' or 'yellow'"


@pytest.mark.parametrize(
    'n, nonumber, s, expected',
    [
        (0, True, '', 'cat'),
        (1, True, '', 'cat'),
        (2, True, '', 'cats'),
        (0, False, '', 'no cat'),
        (1, False, '', '1 cat'),
        (2, False, '', '2 cats'),
        (0, True, ':', 'cat'),
        (1, True, ':', 'cat:'),
        (2, True, ':', 'cats:'),
        (0, False, ':', 'no cat'),
        (1, False, ':', '1 cat:'),
        (2, False, ':', '2 cats:'),
    ],
)
def test_strplural(n, nonumber, s, expected):
    assert strplural(n, 'cat', nonumber=nonumber, s=s) == expected


@pytest.mark.parametrize(
    'broadcast, shape, expected',
    [
        (None, None, 'None'),
        (None, (), '()'),
        (None, (1,), '1'),
        (None, (2, 3), '(2,3)'),
        ('leftward', None, 'None'),
        ('leftward', (), '(...)'),
        ('leftward', (1,), '(...,1)'),
        ('leftward', (2, 3), '(...,2,3)'),
        ('rightward', None, 'None'),
        ('rightward', (), '(...)'),
        ('rightward', (1,), '(1,...)'),
        ('rightward', (2, 3), '(2,3,...)'),
    ],
)
def test_strshape(broadcast, shape, expected):
    assert strshape(shape, broadcast=broadcast) == expected


@pytest.mark.parametrize('value', [1, object(), [1]])
def test_strshape_error(value):
    with pytest.raises(TypeError):
        strshape(value)


def test_timer1():
    t = Timer()
    t0 = time.time()
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    overhead = time.time() - t0 - 0.02

    time.sleep(0.01)
    delta3 = t.elapsed
    assert abs(delta1 - 0.01) < overhead
    assert abs(delta2 - 0.02) < overhead
    assert abs(delta3 - 0.02) < overhead

    t0 = time.time()
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    overhead = time.time() - t0 - 0.02

    time.sleep(0.01)
    delta3 = t.elapsed
    assert abs(delta1 - 0.01) < overhead
    assert abs(delta2 - 0.02) < overhead
    assert abs(delta3 - 0.02) < overhead


def test_timer2():
    t = Timer(cumulative=True)
    t0 = time.time()
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
    overhead1 = time.time() - t0 - 0.01

    time.sleep(0.01)
    delta2 = t.elapsed
    assert abs(delta1 - 0.01) < overhead1
    assert abs(delta2 - 0.01) < overhead1

    t0 = time.time()
    with t:
        time.sleep(0.01)
        delta3 = t.elapsed
        time.sleep(0.01)
        delta4 = t.elapsed
    overhead2 = time.time() - t0 - 0.02

    time.sleep(0.01)
    delta5 = t.elapsed
    assert abs(delta3 - 0.02) < overhead1 + overhead2
    assert abs(delta4 - 0.03) < overhead1 + overhead2
    assert abs(delta5 - 0.03) < overhead1 + overhead2


def test_timer3():
    t = Timer()
    t0 = time.time()
    try:
        with t:
            time.sleep(0.01)
            raise RuntimeError()
    except RuntimeError:
        pass
    overhead = time.time() - t0 - 0.01

    time.sleep(0.01)
    assert t._level == 0
    assert abs(t.elapsed - 0.01) < overhead


def test_zip_broadcast1():
    def g():
        i = 0
        while True:
            yield i
            i += 1

    a = [1]
    b = (np.sin,)
    c = np.arange(3).reshape((1, 3))
    d = ('x', 'y', [])
    e = ['a', 'b', 'c']
    f = np.arange(6).reshape((3, 2))
    h = 3

    aa = []
    bb = []
    cc = []
    dd = []
    ee = []
    ff = []
    gg = []
    hh = []
    for a_, b_, c_, d_, e_, f_, g_, h_ in zip_broadcast(a, b, c, d, e, f, g(), h):
        aa.append(a_)
        bb.append(b_)
        cc.append(c_)
        dd.append(d_)
        ee.append(e_)
        ff.append(f_)
        gg.append(g_)
        hh.append(h_)
    assert aa == 3 * a
    assert bb == list(3 * b)
    assert_equal(cc, 3 * [c[0]])
    assert dd == list(d)
    assert ee == list(e)
    assert_equal(ff, list(f))
    assert gg == [0, 1, 2]
    assert hh == [3, 3, 3]


def test_zip_broadcast2():
    a = [1]
    b = (np.sin,)
    c = np.arange(3).reshape((1, 3))
    aa = []
    bb = []
    cc = []
    for a_, b_, c_ in zip_broadcast(a, b, c):
        aa.append(a_)
        bb.append(b_)
        cc.append(c_)
    assert aa == a
    assert tuple(bb) == b
    assert_equal(cc, list(c))


def test_zip_broadcast3():
    a = 'abc'
    b = [1, 2, 3]
    assert tuple(zip_broadcast(a, b)) == tuple(zip(a, b))
    assert tuple(zip_broadcast(a, b, iter_str=True)) == tuple(zip(a, b))
    assert tuple(zip_broadcast(a, b, iter_str=False)) == (
        ('abc', 1),
        ('abc', 2),
        ('abc', 3),
    )
