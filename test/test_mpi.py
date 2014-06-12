import numpy as np
from pyoperators import (
    IdentityOperator, MPIDistributionGlobalOperator,
    MPIDistributionIdentityOperator, MPI)
from pyoperators.utils import split
from pyoperators.utils.mpi import (
    DTYPE_MAP, OP_PY_MAP, OP_MPI_MAP, as_mpi, combine_shape, distribute_shape,
    distribute_shapes, filter_comm)
from pyoperators.utils.testing import assert_eq, assert_is_type
from numpy.testing import assert_equal

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
dtypes = DTYPE_MAP


def test_allreduce():
    n = 10

    def func(x, xs, op):
        op_py = OP_PY_MAP[op]
        op_mpi = OP_MPI_MAP[op]
        actual = np.empty_like(x)
        comm.Allreduce(as_mpi(x), as_mpi(actual), op=op_mpi)
        expected = op_py(xs)
        assert_equal(actual, expected)
    for dtype in dtypes:
        if dtype.kind in 'ui':
            i = np.iinfo(dtype if dtype != np.uint64 else np.int64)
            x = np.random.random_integers(i.min, i.max-1, size=n).astype(dtype)
        elif dtype.kind == 'f':
            x = np.random.random_integers(-100, 100, size=n).astype(dtype)
        elif dtype.kind == 'c':
            x = np.random.random_integers(-100, 100, size=n) + \
                np.random.random_integers(-100, 100, size=n) * 1j
        else:
            raise TypeError()
        xs = comm.allgather(x)
        for op in OP_PY_MAP:
            if op in ('min', 'max') and dtype.kind == 'c':
                continue
            yield func, x, xs, op


def test_collect():
    def func(comm, s1, s2):
        shape_global = (s1,) + s2
        shape_local = distribute_shape(shape_global, comm=comm)
        shape_global2 = combine_shape(shape_local, comm=comm)
        assert shape_global == shape_global2
    for comm in (MPI.COMM_SELF, MPI.COMM_WORLD):
        for s1 in range(size*2+1):
            for s2 in ((), (2,), (2, 3)):
                yield func, comm, s1, s2


def test_distribute():
    class MyComm(object):
        def __init__(self, rank, size):
            self.rank = rank
            self.size = size
    if size > 1:
        return

    def func(a, r, shape, shapes):
        assert_equal(a[r], shape[0])
        assert_equal(shapes[r], shape)

    for n in range(10):
        for sz in range(1, 7):
            work = np.zeros(n, int)
            for i in range(n):
                work[i] = i % sz
            a = np.zeros(sz, int)
            for r in range(sz):
                a[r] = sum(work == r)
            stop = tuple(np.cumsum(a))
            start = (0,) + stop[:-1]
            comm = MyComm(0, sz)
            for s in [(), (1,), (3, 4)]:
                shapes = distribute_shapes((n,) + s, comm=comm)
                for r in range(sz):
                    shape = distribute_shape((n,) + s, rank=r, size=sz)
                    yield func, a, r, shape, shapes
                    if len(s) > 0:
                        continue
                    sl = slice(start[r], stop[r])
                    yield assert_eq, sl, split(n, sz, r)


def test_dgo():
    def func(shape, dtype):
        d = MPIDistributionGlobalOperator(shape)
        x_global = np.ones(shape, dtype)
        s = split(shape[0], size, rank)
        x_local = d(x_global)
        assert_eq(x_local, x_global[s])
        assert_eq(d.T(x_local), x_global)
    for shape in (2,), (2, 3):
        for dtype in dtypes:
            yield func, shape, dtype


def test_dio():
    def func(shape, dtype):
        x_global = np.ones(shape, dtype)
        d = MPIDistributionIdentityOperator()
        assert_eq(d(x_global), x_global)
        x_local = x_global * (rank + 1)
        assert_eq(d.T(x_local), np.ones(shape) * size * (size + 1) // 2)
    for shape in (2,), (2, 3):
        for dtype in dtypes:
            yield func, shape, dtype


def test_dio_morph():
    op = MPIDistributionIdentityOperator(MPI.COMM_SELF)
    assert_is_type(op, IdentityOperator)


def test_dio_inplace():
    def func(n):
        assert_eq(d.todense(shapein=n), d.todense(shapein=n, inplace=True))
        assert_eq(d.T.todense(shapein=n), d.T.todense(shapein=n, inplace=True))
    d = MPIDistributionIdentityOperator()
    for n in range(10):
        yield func, n


def test_filter_comm():
    comm = MPI.COMM_WORLD

    def func(nglobal):
        d = np.array(comm.rank)
        with filter_comm(comm.rank < nglobal, comm) as newcomm:
            if newcomm is not None:
                newcomm.Allreduce(MPI.IN_PLACE, as_mpi(d))
        d = comm.bcast(d)
        n = min(comm.size, nglobal)
        assert d == n * (n - 1) // 2
    for nglobal in range(comm.size + 3):
        yield func, nglobal
