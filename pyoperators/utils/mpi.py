try:
    from mpi4py import MPI
except ImportError:
    import fake_MPI as MPI
from .misc import tointtuple

__all__ = ['MPI', 'combine_shape', 'distribute_shape', 'distribute_slice']


def combine_shape(shape, comm=None):
    """
    Return the shape of the global array resulting from stacking local arrays
    along the first dimension.

    """
    shape = tointtuple(shape)
    comm = comm or MPI.COMM_WORLD
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
            raise ValueError('It is ambiguous to split a scalar across processe' 's.')
        return ()
    nglobal = shape[0]
    nlocal = nglobal // size + ((nglobal % size) > rank)
    return (nlocal,) + tuple(shape[1:])


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
