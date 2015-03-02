import itertools
import numpy as np
import os
import time

from contextlib import contextmanager
from numpy.testing import assert_equal, assert_warns
from pyoperators import Operator
from pyoperators.utils import (
    broadcast_shapes, cast, complex_dtype, complex_intrinsic_dtype, deprecated,
    first, first_is_not, float_dtype, float_intrinsic_dtype,
    float_or_complex_dtype, ifirst, ifirst_is_not, ilast, ilast_is_not,
    groupbykey, inspect_special_values, interruptible, isscalarlike,
    zip_broadcast, last, last_is_not, least_greater_multiple, one,
    omp_num_threads, pi, pool_threading, product, reshape_broadcast, setting,
    settingerr, split, strenum, strplural, strshape, Timer, uninterruptible,
    zero)
from pyoperators.utils.testing import (
    assert_eq, assert_not_in, assert_raises, assert_same)
from pyoperators.warnings import PyOperatorsDeprecationWarning

dtypes = [np.dtype(t) for t in (np.bool8, np.uint8, np.int8, np.uint16,
          np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32,
          np.float64, np.float128, np.complex64, np.complex128, np.complex256)]


def assert_dtype(a, d):
    if a is None:
        return
    assert_eq(a.dtype, d)


def test_deprecated():
    class mynewclass(object):
        def __init__(self, a, b=None):
            self.a = a
            self.b = b

    assert_raises(TypeError, deprecated, lambda x: x)


    @deprecated('blabla')
    def myfunc():
        return 2


    @deprecated
    class myclass1(mynewclass):
        pass


    @deprecated('blabla2')
    class myclass2(mynewclass):
        pass

    val = assert_warns(PyOperatorsDeprecationWarning, myfunc)
    assert_equal(val, 2)

    c = assert_warns(PyOperatorsDeprecationWarning, myclass1, 3, b=2)
    assert_equal(c.a, 3)
    assert_equal(c.b, 2)

    c = assert_warns(PyOperatorsDeprecationWarning, myclass2, 3, b=2)
    assert_equal(c.a, 3)
    assert_equal(c.b, 2)


def test_broadcast_shapes():
    shapes = [((),), ((), ()), ((), (), ()),
              ((1,),), ((), (1,)), ((), (1,), (1,)),
              ((2,),), ((), (2,)), ((1,), (2,)),
              ((), (2,), (2,)), ((), (2,), (1,)), ((2,), (1,), ()),
              ((1,), (2, 1)), ((), (2, 1)), ((1,), (2, 1), ()),
              ((1,), (1, 2)), ((), (1, 2)), ((1,), (1, 2), ()),
              ((2,), (2, 1)), ((), (2,), (2, 1)), ((2,), (2, 1), ()),
              ((1, 2), (2, 1)), ((), (1, 2), (2, 1)), ((1, 2), (2, 1), ()),
              ((1, 1, 4), (1, 3, 1), (2, 1, 1), (), (1, 1, 1))]
    expecteds = [(), (), (),
                 (1,), (1,), (1,),
                 (2,), (2,), (2,),
                 (2,), (2,), (2,),
                 (2, 1), (2, 1), (2, 1),
                 (1, 2), (1, 2), (1, 2),
                 (2, 2), (2, 2), (2, 2),
                 (2, 2), (2, 2), (2, 2),
                 (2, 3, 4)]

    def func(shape, expected):
        assert_equal(broadcast_shapes(*shape), expected)
    for shape, expected in zip(shapes, expecteds):
        yield func, shape, expected


def test_cast():
    dtypes = (float, complex, None)

    def func(d1, d2):
        a1 = None if d1 is None else np.array(1, dtype=d1)
        a2 = None if d2 is None else np.array(1, dtype=d2)
        if d1 is None and d2 is None:
            assert_raises(ValueError, cast, [a1, a2])
            return
        expected = d1 if d2 is None else d2 if d1 is None else \
            np.promote_types(d1, d2)
        a1_, a2_ = cast([a1, a2])
        assert_dtype(a1_, expected)
        assert_dtype(a2_, expected)
    for d1, d2 in itertools.product(dtypes, repeat=2):
        yield func, d1, d2


def test_complex_dtype():
    dtypes = (str, bool, int, np.uint32,
              np.float16, np.float32, np.float64, np.float128,
              np.complex64, np.complex128, np.complex256,
              '>f2', '>f4', '>f8', '>f16',
              '>c8', '>c16', '>c32')
    expecteds = (None, '<c16', '<c16', '<c16',
                 '<c16', '<c8', '<c16', '<c32',
                 '<c8', '<c16', '<c32',
                 '<c16', '>c8', '>c16', '>c32',
                 '>c8', '>c16', '>c32')

    def func(dtype, expected):
        if expected is None:
            assert_raises(TypeError, complex_dtype, dtype)
        else:
            actual = complex_dtype(dtype)
            assert_eq(actual.str, expected)
    for dtype, expected in zip(dtypes, expecteds):
        yield func, dtype, expected


def test_complex_intrinsic_dtype():
    dtypes = (str, bool, int, np.uint32,
              np.float16, np.float32, np.float64, np.float128,
              np.complex64, np.complex128, np.complex256,
              '>f2', '>f4', '>f8', '>f16',
              '>c8', '>c16', '>c32')
    expecteds = (None, '<c16', '<c16', '<c16',
                 '<c8', '<c8', '<c16', '<c16',
                 '<c8', '<c16', '<c16',
                 '<c8', '<c8', '<c16', '<c16',
                 '<c8', '<c16', '<c16')

    def func(dtype, expected):
        if expected is None:
            assert_raises(TypeError, complex_dtype, dtype)
        else:
            actual = complex_intrinsic_dtype(dtype)
            assert_eq(actual.str, expected)
    for dtype, expected in zip(dtypes, expecteds):
        yield func, dtype, expected


def test_float_dtype():
    dtypes = (str, bool, int, np.uint32,
              np.float16, np.float32, np.float64, np.float128,
              np.complex64, np.complex128, np.complex256,
              '>f2', '>f4', '>f8', '>f16',
              '>c8', '>c16', '>c32')
    expecteds = (None, '<f8', '<f8', '<f8',
                 '<f2', '<f4', '<f8', '<f16',
                 '<f4', '<f8', '<f16',
                 '>f2', '>f4', '>f8', '>f16',
                 '>f4', '>f8', '>f16')

    def func(dtype, expected):
        if expected is None:
            assert_raises(TypeError, float_dtype, dtype)
        else:
            actual = float_dtype(dtype)
            assert_eq(actual, expected)
    for dtype, expected in zip(dtypes, expecteds):
        yield func, dtype, expected


def test_float_intrinsic_dtype():
    dtypes = (str, bool, int, np.uint32,
              np.float16, np.float32, np.float64, np.float128,
              np.complex64, np.complex128, np.complex256,
              '>f2', '>f4', '>f8', '>f16',
              '>c8', '>c16', '>c32')
    expecteds = (None, '<f8', '<f8', '<f8',
                 '<f4', '<f4', '<f8', '<f8',
                 '<f4', '<f8', '<f8',
                 '<f4', '<f4', '<f8', '<f8',
                 '<f4', '<f8', '<f8')

    def func(dtype, expected):
        if expected is None:
            assert_raises(TypeError, complex_dtype, dtype)
        else:
            actual = float_intrinsic_dtype(dtype)
            assert_eq(actual.str, expected)
    for dtype, expected in zip(dtypes, expecteds):
        yield func, dtype, expected


def test_float_or_complex_dtype():
    dtypes = (str, bool, int, np.uint32,
              np.float16, np.float32, np.float64, np.float128,
              np.complex64, np.complex128, np.complex256,
              '>f2', '>f4', '>f8', '>f16',
              '>c8', '>c16', '>c32')
    expecteds = (None, '<f8', '<f8', '<f8',
                 '<f2', '<f4', '<f8', '<f16',
                 '<c8', '<c16', '<c32',
                 '>f2', '>f4', '>f8', '>f16',
                 '>c8', '>c16', '>c32')

    def func(dtype, expected):
        if expected is None:
            assert_raises(TypeError, float_dtype, dtype)
        else:
            actual = float_or_complex_dtype(dtype)
            assert_eq(actual, expected)
    for dtype, expected in zip(dtypes, expecteds):
        yield func, dtype, expected


def test_first1():
    assert first([1, 2, 3], lambda x: x > 1.5) == 2
    assert last([1, 2, 3], lambda x: x > 1.5) == 3


def test_first2():
    assert_raises(ValueError, first, [1, 2, 3], lambda x: x > 3)
    assert_raises(ValueError, last, [1, 2, 3], lambda x: x > 3)


def test_ifirst1():
    assert ifirst([1, 2, 3], lambda x: x > 1.5) == 1
    assert ilast([1, 2, 3], lambda x: x > 1.5) == 2


def test_ifirst2():
    assert_raises(ValueError, ifirst, [1, 2, 3], lambda x: x > 3)
    assert_raises(ValueError, ilast, [1, 2, 3], lambda x: x > 3)


def test_ifirst3():
    assert ifirst([1, 2, 2, 3], 2.) == 1
    assert ilast([1, 2, 2, 3], 2.) == 2


def test_ifirst4():
    assert_raises(ValueError, ifirst, [1, 2, 3], 4)
    assert_raises(ValueError, ilast, [1, 2, 3], 4)


def test_first_is_not():
    assert first_is_not([1, 2], 1) == 2
    assert first_is_not([None, None, {}], None) == {}
    assert first_is_not([], None) is None
    assert first_is_not([None, None], None) is None

    assert last_is_not([1, 2], 2) == 1
    assert last_is_not([{}, None, None], None) == {}
    assert last_is_not([], None) is None
    assert last_is_not([None, None], None) is None


def test_ifirst_is_not():
    assert ifirst_is_not([1, 2, 2], 2) == 0
    assert ifirst_is_not([2, 1, 1], 2) == 1
    assert ifirst_is_not([{}, None, None], None) == 0
    assert_raises(ValueError, ifirst_is_not, [], None)
    assert_raises(ValueError, ifirst_is_not, [None, None], None,)

    assert ilast_is_not([1, 2, 2], 2) == 0
    assert ilast_is_not([2, 1, 1], 2) == 2
    assert ilast_is_not([{}, None, None], None) == 0
    assert_raises(ValueError, ilast_is_not, [], None)
    assert_raises(ValueError, ilast_is_not, [None, None], None,)


def test_groupbykey():
    vals = ['a', 'b', 'c', 'd']
    keys = itertools.combinations_with_replacement([1, 2, 3, 4], 4)

    def func(k):
        result = list(groupbykey(vals, k))
        expected = [(k, tuple(i[0] for i in it)) for k, it in
                    itertools.groupby(zip(vals, k), lambda x: x[1])]
        assert_equal(result, expected)
    for k in keys:
        yield func, k


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
        for x in ((1, 1.1, 0, -1, -1), (-1, -1), (0, 0), (1, 1), (2, 2),
                  (2, 1), np.random.random_integers(-1, 1, size=10)):
            x = np.asarray(x).astype(d)
            yield func, x


def test_interruptible():
    import signal

    def func_interruptible():
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def func_uninterruptible():
        assert signal.getsignal(signal.SIGINT) is not \
            signal.default_int_handler

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


def test_is_scalar():
    def func(x):
        assert isscalarlike(x)
    for x in (True, 1, 1., 'lkj', 'jj', np.array(1)):
        yield func, x


def test_is_not_scalar():
    def func(x):
        assert not isscalarlike(x)
    for x in ([], (), np.ones((0, 1)), np.ones(1), np.ones(2), object, np.int8,
              slice, Operator()):
        yield func, x


def test_least_greater_multiple():
    def func(lgm, expected):
        assert_eq(lgm, expected)
    a, b, c = np.ogrid[[slice(4, 11) for m in range(3)]]
    expected = 2**a * 3**b * 5**c
    yield func, least_greater_multiple(expected, [2, 3, 5]), expected
    yield func, least_greater_multiple(expected-1, [2, 3, 5]), expected


def test_one_pi_zero():
    expected = 1, 4 * np.arctan(np.array(1, np.float128)), 0

    def func(f, dtype, exp):
        assert_same(f(dtype), np.array(exp, dtype=dtype))
    for f, exp in zip((one, pi, zero), expected):
        for dtype in (np.float16, np.float32, np.float64, np.float128,
                      np.complex64, np.complex128, np.complex256):
            yield func, f, dtype, exp


def test_pool_threading():
    try:
        import mkl
        mkl_nthreads = mkl.get_max_threads()
    except ImportError:
        mkl = None
    counter = None

    def func_thread(i):
        global counter
        counter += 1

    @contextmanager
    def get_env(value):
        try:
            del os.environ['OMP_NUM_THREADS']
        except KeyError:
            pass
        if value is not None:
            os.environ['OMP_NUM_THREADS'] = str(value)
        yield
        if value is not None:
            del os.environ['OMP_NUM_THREADS']

    def func(env):
        global counter
        with env:
            nthreads = os.getenv('OMP_NUM_THREADS')
            expected = omp_num_threads()
            with pool_threading() as pool:
                assert_equal(int(os.environ['OMP_NUM_THREADS']), 1)
                if mkl is not None:
                    assert_equal(mkl.get_max_threads(), 1)
                counter = 0
                pool.map(func_thread, range(pool._processes))
            assert_equal(os.getenv('OMP_NUM_THREADS'), nthreads)
            if mkl is not None:
                assert_equal(mkl.get_max_threads(), mkl_nthreads)
            assert_equal(counter, expected)
        assert_not_in('OMP_NUM_THREADS', os.environ)

    for env in get_env(None), get_env(1), get_env(3):
        yield func, env


def test_product():
    def func(o):
        assert o == 1
    for o in ([], (), (1,), [1], [2, 0.5], (2, 0.5), np.array(1),
              np.array([2, 0.5])):
        yield func, product(o)


def test_reshape_broadcast():
    data = np.arange(20)
    shapes = (4, 5), (1, 4, 5), (4, 1, 5), (4, 5, 1)
    new_shapess = (
        ((4, 5), (1, 4, 5), (2, 4, 5), (2, 3, 4, 5)),
        ((1, 4, 5), (2, 4, 5), (1, 2, 4, 5), (2, 2, 4, 5), (2, 3, 2, 4, 5)),
        ((4, 1, 5), (4, 2, 5), (1, 4, 2, 5), (2, 4, 2, 5), (2, 3, 4, 2, 5)),
        ((4, 5, 1), (4, 5, 2), (1, 4, 5, 2), (2, 4, 5, 2), (2, 3, 4, 5, 2)))

    def func(shape, new_shape):
        data_ = data.reshape(shape)
        expected = np.empty(new_shape)
        expected[...] = data_
        actual = reshape_broadcast(data_, new_shape)
        assert_equal(actual, expected)
    for shape, new_shapes in zip(shapes, new_shapess):
        for new_shape in new_shapes:
            yield func, shape, new_shape


def test_setting():
    class Obj():
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
    ref2 = {'divide': 'ignore', 'invalid': 'ignore', 'over': 'ignore',
            'under': 'ignore'}
    ref3 = {'divide': 'raise', 'invalid': 'ignore', 'over': 'warn',
            'under': 'ignore'}

    with settingerr(all='ignore'):
        assert_eq(np.seterr(), ref2)
        with settingerr(divide='raise', over='warn'):
            assert_eq(np.seterr(), ref3)
        assert_eq(np.seterr(), ref2)
    assert_eq(np.seterr(), ref1)


def test_split():
    def func(n, m):
        slices = tuple(split(n, m))
        assert_eq(len(slices), m)
        x = np.zeros(n, int)
        for s in slices:
            x[s] += 1
        assert_same(x, 1, broadcasting=True)
        assert_eq([split(n, m, i) for i in range(m)], slices)
    for n in range(4):
        for m in range(1, 6):
            yield func, n, m


def test_strenum():
    assert_eq(strenum(['blue', 'red', 'yellow'], 'or'),
              "'blue', 'red' or 'yellow'")


def test_strplural():
    def func(n, nonumber, s, expected):
        assert_eq(strplural(n, 'cat', nonumber=nonumber, s=s), expected)
    for n, nonumber, s, expected in zip(
            4*(0, 1, 2),
            3*(True,) + 3*(False,) + 3*(True,) + 3*(False,),
            6*('',) + 6*(':',),
            ('cat', 'cat', 'cats', 'no cat', '1 cat', '2 cats',
             'cat', 'cat:', 'cats:', 'no cat', '1 cat:', '2 cats:')):
        yield func, n, nonumber, s, expected


def test_strshape():
    shapes = (None, (), (1,), (2, 3))
    broadcasts = None, 'leftward', 'rightward'
    expectedss = [('None', '()', '1', '(2,3)'),
                  ('None', '(...)', '(...,1)', '(...,2,3)'),
                  ('None', '(...)', '(1,...)', '(2,3,...)')]

    def func(shape, broadcast, expected):
        assert_equal(strshape(shape, broadcast=broadcast), expected)
    for broadcast, expecteds in zip(broadcasts, expectedss):
        for shape, expected in zip(shapes, expecteds):
            yield func, shape, broadcast, expected


def test_strshape_error():
    def func(x):
        assert_raises(TypeError, strshape, x)
    for x in 1, object(), [1]:
        yield func, x


def test_timer1():
    t = Timer()
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    time.sleep(0.01)
    delta3 = t.elapsed
    assert abs(delta1 - 0.01) < 0.001
    assert abs(delta2 - 0.02) < 0.001
    assert abs(delta3 - 0.02) < 0.001
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    time.sleep(0.01)
    delta3 = t.elapsed
    assert abs(delta1 - 0.01) < 0.001
    assert abs(delta2 - 0.02) < 0.001
    assert abs(delta3 - 0.02) < 0.001


def test_timer2():
    t = Timer(cumulative=True)
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    time.sleep(0.01)
    delta3 = t.elapsed

    assert abs(delta1 - 0.01) < 0.001
    assert abs(delta2 - 0.02) < 0.001
    assert abs(delta3 - 0.02) < 0.001
    with t:
        time.sleep(0.01)
        delta1 = t.elapsed
        time.sleep(0.01)
        delta2 = t.elapsed
    time.sleep(0.01)
    delta3 = t.elapsed
    assert abs(delta1 - 0.03) < 0.001
    assert abs(delta2 - 0.04) < 0.001
    assert abs(delta3 - 0.04) < 0.001


def test_timer3():
    t = Timer()
    try:
        with t:
            time.sleep(0.01)
            raise RuntimeError()
    except RuntimeError:
        pass
    time.sleep(0.01)
    assert_equal(t._level, 0)
    assert abs(t.elapsed - 0.01) < 0.001


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

    aa = []; bb = []; cc = []; dd = []; ee = []; ff = []; gg = []; hh = []
    for a_, b_, c_, d_, e_, f_, g_, h_ in zip_broadcast(
            a, b, c, d, e, f, g(), h):
        aa.append(a_)
        bb.append(b_)
        cc.append(c_)
        dd.append(d_)
        ee.append(e_)
        ff.append(f_)
        gg.append(g_)
        hh.append(h_)
    assert_eq(aa, 3 * a)
    assert_eq(bb, list(3 * b))
    assert_eq(cc, [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    assert_eq(dd, list(_ for _ in d))
    assert_eq(ee, list(_ for _ in e))
    assert_eq(ff, list(_ for _ in f))
    assert_eq(gg, [0, 1, 2])
    assert_eq(hh, [3, 3, 3])


def test_zip_broadcast2():
    a = [1]
    b = (np.sin,)
    c = np.arange(3).reshape((1, 3))
    aa = []; bb = []; cc = []
    for a_, b_, c_ in zip_broadcast(a, b, c):
        aa.append(a_)
        bb.append(b_)
        cc.append(c_)
    assert_eq(aa, a)
    assert_eq(tuple(bb), b)
    assert_eq(cc, c)


def test_zip_broadcast3():
    a = 'abc'
    b = [1, 2, 3]
    assert_eq(tuple(zip_broadcast(a, b)), tuple(zip(a, b)))
    assert_eq(tuple(zip_broadcast(a, b, iter_str=True)), tuple(zip(a, b)))
    assert_eq(tuple(zip_broadcast(a, b, iter_str=False)),
              (('abc', 1), ('abc', 2), ('abc', 3)))
