import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import (
    MPI,
    IdentityOperator,
    MPIDistributionGlobalOperator,
    MPIDistributionIdentityOperator,
)
from pyoperators.utils import split
from pyoperators.utils.mpi import (
    DTYPE_MAP,
    OP_MPI_MAP,
    OP_PY_MAP,
    as_mpi,
    combine_shape,
    distribute_shape,
    distribute_shapes,
    filter_comm,
)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
dtypes = DTYPE_MAP


@pytest.mark.parametrize('dtype', DTYPE_MAP)
def test_allreduce(dtype):
    n = 10

    def assert_allreduce(x, xs, op):
        op_py = OP_PY_MAP[op]
        op_mpi = OP_MPI_MAP[op]
        actual = np.empty_like(x)
        comm.Allreduce(as_mpi(x), as_mpi(actual), op=op_mpi)
        expected = op_py(xs)
        assert_equal(actual, expected)

    if dtype.kind in 'ui':
        i = np.iinfo(dtype if dtype != np.uint64 else np.int64)
        x = np.random.randint(i.min, i.max, size=n).astype(dtype)
    elif dtype.kind == 'f':
        x = np.random.randint(-100, 101, size=n).astype(dtype)
    elif dtype.kind == 'c':
        x = (
            np.random.randint(-100, 101, size=n)
            + np.random.randint(-100, 101, size=n) * 1j
        )
    else:
        raise TypeError()

    xs = comm.allgather(x)
    for op in OP_PY_MAP:
        if op in ('min', 'max') and dtype.kind == 'c':
            continue
        assert_allreduce(x, xs, op)


@pytest.mark.parametrize('comm', [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize('s1', range(size * 2 + 1))
@pytest.mark.parametrize('s2', [(), (2,), (2, 3)])
def test_collect(comm, s1, s2):
    shape_global = (s1,) + s2
    shape_local = distribute_shape(shape_global, comm=comm)
    shape_global2 = combine_shape(shape_local, comm=comm)
    assert shape_global == shape_global2


@pytest.mark.parametrize('n', range(10))
@pytest.mark.parametrize('sz', range(1, 7))
def test_distribute(n, sz):
    class MyComm:
        def __init__(self, rank, size):
            self.rank = rank
            self.size = size

    if size > 1:
        return

    def func(a, r, shape, shapes):
        assert a[r] == shape[0]
        assert shapes[r] == shape

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
            func(a, r, shape, shapes)
            if len(s) > 0:
                continue
            sl = slice(start[r], stop[r])
            assert_equal(sl, split(n, sz, r))


@pytest.mark.parametrize('shape', [(2,), (2, 3)])
@pytest.mark.parametrize('dtype', DTYPE_MAP)
def test_dgo(shape, dtype):
    d = MPIDistributionGlobalOperator(shape)
    x_global = np.ones(shape, dtype)
    s = split(shape[0], size, rank)
    x_local = d(x_global)
    assert_equal(x_local, x_global[s])
    assert_equal(d.T(x_local), x_global)


@pytest.mark.parametrize('shape', [(2,), (2, 3)])
@pytest.mark.parametrize('dtype', DTYPE_MAP)
def test_dio(shape, dtype):
    x_global = np.ones(shape, dtype)
    d = MPIDistributionIdentityOperator()
    assert_equal(d(x_global), x_global)
    x_local = x_global * (rank + 1)
    assert_equal(d.T(x_local), np.ones(shape) * size * (size + 1) // 2)


def test_dio_morph():
    op = MPIDistributionIdentityOperator(MPI.COMM_SELF)
    assert type(op) is IdentityOperator


@pytest.mark.parametrize('n', range(10))
def test_dio_inplace(n):
    d = MPIDistributionIdentityOperator()
    assert_equal(d.todense(shapein=n), d.todense(shapein=n, inplace=True))
    assert_equal(d.T.todense(shapein=n), d.T.todense(shapein=n, inplace=True))


@pytest.mark.parametrize('nglobal', range(comm.size + 3))
def test_filter_comm(nglobal):
    comm = MPI.COMM_WORLD
    d = np.array(comm.rank)
    with filter_comm(comm.rank < nglobal, comm) as newcomm:
        if newcomm is not None:
            newcomm.Allreduce(MPI.IN_PLACE, as_mpi(d))
    d = comm.bcast(d)
    n = min(comm.size, nglobal)
    assert d == n * (n - 1) // 2
