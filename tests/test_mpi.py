import numpy as np
from nose.plugins.skip import SkipTest
from numpy.testing import assert_equal
from pyoperators.mpi_operators import (DistributionGlobalOperator,
     DistributionIdentityOperator, distribute_shape, distribute_slice)

try:
    from mpi4py import MPI
except ImportError:
    raise SkipTest

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
dtypes = np.int8, np.int32, np.int64, np.float32, np.float64

def test_distribute():
    if size > 1:
        return
    for n in range(10):
        for sz in range(1,7):
            work = np.zeros(n, int)
            for i in range(n):
                work[i] = i % sz
            a = np.zeros(sz, int)
            for r in range(sz):
                a[r] = sum(work==r)
            stop = tuple(np.cumsum(a))
            start = (0,) + stop[:-1]
            for r in range(sz):
                yield assert_equal, a[r], distribute_shape((n,), rank=r,
                                                           size=sz)[0]
                s = slice(start[r], stop[r])
                yield assert_equal, s, distribute_slice(n, rank=r, size=sz)

def test_dgo():
    def func(shape, dtype):
        d = DistributionGlobalOperator(shape)
        x_global = np.ones(shape)
        s = distribute_slice(shape[0])
        x_local = d(x_global)
        assert_equal(x_local, x_global[s])
        assert_equal(d.T(x_local), x_global)
    for shape in (2,), (2,3):
        for dtype in dtypes:
            yield func, shape, dtype

def test_dio():
    def func(shape, dtype):
        x_global = np.ones(shape)
        d = DistributionIdentityOperator()
        assert_equal(d(x_global), x_global)
        x_local = x_global * (rank + 1)
        assert_equal(d.T(x_local), np.ones(shape) * size * (size + 1) // 2)
    for shape in (2,), (2,3):
        for dtype in dtypes:
            yield func, shape, dtype

def test_dio_inplace():
    def func(n):
        assert_equal(d.todense(shapein=n), d.todense(shapein=n, inplace=True))
        assert_equal(d.T.todense(shapein=n), d.T.todense(shapein=n,
                                                         inplace=True))
    d = DistributionIdentityOperator()
    for n in range(10):
        yield func, n
