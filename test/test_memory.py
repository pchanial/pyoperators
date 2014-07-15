from __future__ import division

import itertools
import numpy as np
from numpy.testing import assert_equal
from pyoperators.config import MEMORY_ALIGNMENT
from pyoperators.memory import MemoryPool, empty
from pyoperators.utils import tointtuple

buffers = [empty(10), empty((5, 2)), empty(20)[::2], empty(11)[1:],
           empty(21)[1:].reshape((10, 2))[::2, :]]
aligned = 3 * [True] + [False, False]
contiguous = [_.flags.contiguous for _ in buffers]


def assert_contiguous(x):
    assert x.flags.contiguous


def assert_aligned(x):
    assert address(x) % MEMORY_ALIGNMENT == 0


def address(l):
    if isinstance(l, np.ndarray):
        return l.__array_interface__['data'][0]
    return [address(_) for _ in l]


def test_empty():
    shapes = (10, (10,), (2, 10), (3, 3, 3))
    dtypes = (float, np.int8, complex)

    def func(v, s, d):
        assert_equal(v.shape, tointtuple(s))
        assert_equal(v.dtype, d)
        assert_aligned(v)
        assert_contiguous(v)
    for s in shapes:
        for d in dtypes:
            v = empty(s, d)
            yield func, v, s, d


def test_set():
    pool = MemoryPool()
    a = np.empty(9)
    c = np.empty(11)
    pool.add(a)
    pool.add(c)

    def func(b):
        assert address(pool._buffers) == address([a, b, c])
    for b in buffers:
        with pool.set(b):
            yield func, b
        assert address(pool._buffers) == address([a, c])


def test_get():
    pool = MemoryPool()
    pa = empty(9)
    pc = empty(11)
    pool.add(pa)
    pool.add(pc)

    def func(v, b, bs, ba, bc, s, a, c):
        assert_equal(v.shape, s)
        if a:
            assert_aligned(v)
        if c:
            assert_contiguous(v)
        if a > ba or c and not bc or not bc and s != bs:
            assert address(pool._buffers) == address([pa, b])
        else:
            assert address(pool._buffers) == address([pa, pc])
    for b, ba, bc in zip(buffers, aligned, contiguous):
        with pool.set(b):
            for (s, a, c) in itertools.product([(10,), (5, 2), (2, 5)],
                                               [False, True],
                                               [False, True]):
                with pool.get(s, float, a, c) as v:
                    yield func, v, b, b.shape, ba, bc, s, a, c
                assert address(pool._buffers) == address([pa, b, pc])
        assert address(pool._buffers) == address([pa, pc])


def test_new_entry():
    pool = MemoryPool()
    a = empty(12)
    b = empty(20)
    pool.add(a)
    pool.add(b)
    shapes = ((4,), (15,), (30,))

    def func(i, s, d=-1):
        assert_equal(len(pool), 3 + i)
    for i, s in enumerate(shapes):
        with pool.get(s, float):
            pass
        yield func, i, s
    for s in [a.shape, b.shape]:
        for d in [0, 1, 2]:
            with pool.get(s[0] - d, float):
                pass
            yield func, i, s, d
