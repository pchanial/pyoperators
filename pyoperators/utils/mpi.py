
import contextlib
import numpy as np
import operator
import os
from .. import config
from functools import reduce

try:
    if config.NO_MPI:
        raise ImportError()
    from mpi4py import MPI
except ImportError:
    from . import fake_MPI as MPI
from .misc import deprecated, isscalarlike, Timer, tointtuple

__all__ = ['MPI',
           'as_mpi',
           'combine',
           'distribute',
           'combine_shape',
           'distribute_shape',
           'distribute_slice',
           'filter_comm',
           'mprint']

DTYPE_MAP = {
    np.dtype(np.int8): MPI.SIGNED_CHAR,
    np.dtype(np.int16): MPI.SHORT,
    np.dtype(np.int32): MPI.INT,
    np.dtype(np.int64): MPI.LONG,
    np.dtype(np.uint8): MPI.UNSIGNED_CHAR,
    np.dtype(np.uint16): MPI.UNSIGNED_SHORT,
    np.dtype(np.uint32): MPI.UNSIGNED_INT,
    np.dtype(np.uint64): MPI.UNSIGNED_LONG,
    np.dtype(np.float32): MPI.FLOAT,
    np.dtype(np.float64): MPI.DOUBLE,
    np.dtype(np.complex64): MPI.COMPLEX,
    np.dtype(np.complex128): MPI.DOUBLE_COMPLEX,
}

IOP_PY_MAP = {'sum':operator.iadd,
              'prod':operator.imul,
              'min':lambda x,y:np.minimum(x,y,x),
              'max':lambda x,y:np.maximum(x,y,x)}
OP_PY_MAP = {'sum':sum,
             'prod':lambda x: reduce(np.multiply, x),
             'min':lambda x: reduce(np.minimum, x),
             'max':lambda x: reduce(np.maximum, x)}
OP_MPI_MAP = {'sum':MPI.SUM,
              'prod':MPI.PROD,
              'min':MPI.MIN,
              'max':MPI.MAX}

timer_mpi = Timer(cumulative=True)


def as_mpi(x):
    try:
        return x, DTYPE_MAP[x.dtype]
    except KeyError:
        raise KeyError("The dtype '{0}' is not handled in MPI.".format(
                       x.dtype.name))


def combine(n, comm=MPI.COMM_WORLD):
    """
    Return total number of work items.
    """
    n = np.array(n)
    with timer_mpi:
        comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
    return int(n)


@deprecated("use 'split' instead.")
def distribute(n, comm=MPI.COMM_WORLD):
    """
    Distribute work across processors.
    """
    if isscalarlike(n):
        return n // comm.size + ((n % comm.size) > comm.rank)
    n = np.asanyarray(n)
    s = distribute_slice(n.shape[0], comm=comm)
    return n[s]


def combine_shape(shape, comm=None):
    """
    Return the shape of the global array resulting from stacking local arrays
    along the first dimension.

    """
    shape = tointtuple(shape)
    comm = comm or MPI.COMM_WORLD
    with timer_mpi:
        shapes = comm.allgather(shape)
    if any(len(s) != len(shapes[0]) or s[1:] != shapes[0][1:] for s in shapes):
        raise ValueError("The shapes are incompatible: '{0}'.".format(shapes))
    return (sum(s[0] for s in shapes),) + shapes[0][1:]


def distribute_shape(shape, rank=None, size=None, comm=None):
    """
    Return the shape of a local array given the shape of a global array,
    according to the rank of the MPI job, The load is distributed along
    the first dimension.
    """
    from .misc import tointtuple

    if rank is None or size is None:
        comm = comm or MPI.COMM_WORLD
    if size is None:
        size = comm.size
    if rank is None:
        rank = comm.rank

    shape = tointtuple(shape)
    if len(shape) == 0:
        if size > 1:
            raise ValueError(
                'It is ambiguous to split a scalar across processes.')
        return ()
    nglobal = shape[0]
    nlocal = nglobal // size + ((nglobal % size) > rank)
    return (nlocal,) + tuple(shape[1:])


def distribute_shapes(shape, comm=None):
    """
    Return the list of the local array shapes given the shape of a global
    array, for all MPI processes. The load is distributed along the first
    dimension.

    """
    if comm is None:
        comm = MPI.COMM_WORLD
    size = comm.size
    nglobal = shape[0]
    shape_first = (nglobal // size + 1,) + shape[1:]
    shape_last = (nglobal // size,) + shape[1:]
    nfirst = nglobal % size
    return nfirst * (shape_first,) + (size-nfirst) * (shape_last,)


@deprecated("use 'split' instead.")
def distribute_slice(nglobal, rank=None, size=None, comm=None):
    """
    Given a number of ordered global work items, return the slice that brackets
    the items distributed to a local MPI job.

    """
    if rank is None or size is None:
        comm = comm or MPI.COMM_WORLD
    if size is None:
        size = comm.size
    if rank is None:
        rank = comm.rank
    nlocal = nglobal // size + ((nglobal % size) > rank)
    start = nglobal // size * rank + min(rank, nglobal % size)
    stop = start + nlocal
    return slice(start, stop)


@contextlib.contextmanager
def filter_comm(condition, comm):
    """
    Return a context manager whose return value is a communicator that only
    include processes for which the specified condition is met or None
    otherwise.

    Parameters:
    -----------
    condition : boolean
        Condition to be met to include the process in the new communicator.
    comm : mpi4py.MPI.Comm
        The communicator of the processes that reach the execution of
        this function. These processes will be included in the new communicator
        if condition is True.
        
    Example:
    --------
    The following snippet prints the list of the rank of the 3 first processes,
    for any number of MPI processes greater than 3:
    with filter_comm(comm.rank < 3, MPI.COMM_WORLD) as newcomm:
        if newcomm is not None:
            print(newcomm.allgather(newcomm.rank))

    """
    with timer_mpi:
        newcomm = comm.Split(color=int(condition), key=comm.rank)
    if not condition:
        yield None
    else:
        yield newcomm
    with timer_mpi:
        newcomm.Free()


def mprint(msg='', comm=MPI.COMM_WORLD):
    """
    Print message on stdout. If the message is the same for all nodes,
    only print one message. Otherwise, add rank information.

    All messages are gathered and printed by rank 0 process, to make sure that
    messages are printed in rank order.

    """
    msgs = comm.gather(msg)
    if comm.rank == 0:
        if all(m == msgs[0] for m in msgs):
            print(msg)
        else:
            print('\n'.join('Rank {}: {}'.format(i, m)
                            for i, m in enumerate(msgs)))
