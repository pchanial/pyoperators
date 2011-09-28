"""
This module handles the allocation of memory.
"""
from __future__ import division

import gc
import numpy as np
from . import utils

__all__ = []

verbose = True
stack = []
scratch = None


def allocate(shape, dtype, buf, description):
    """
    Return an array of given shape and dtype. If a buffer is provided and
    is large enough, it is reused, otherwise a memory allocation takes
    place. Every allocation should go through this method.
    """

    if utils.isscalar(shape):
        shape = (shape,)
    dtype = np.dtype(dtype)

    requested = dtype.itemsize * np.product(shape)
    if buf is not None and buf.nbytes >= requested:
        if utils.isscalar(buf):
            buf = buf.reshape(1)
        buf = buf.view(np.int8).ravel()[:requested].view(dtype).reshape(shape)
        return wrap_ndarray(buf), False

    if verbose:
        if requested < 1024:
            snbytes = str(requested) + ' bytes'
        else:
            snbytes = str(requested / 2**20) + ' MiB'
        print(
            'Info: Allocating '
            + str(shape).replace(' ', '')
            + ' '
            + dtype.type.__name__
            + ' = '
            + snbytes
            + ' in '
            + description
            + '.'
        )
    try:
        buf = np.empty(shape, dtype)
    except MemoryError:
        gc.collect()
        buf = np.empty(shape, dtype)

    return wrap_ndarray(buf), True


def allocate_like(a, b, description):
    """Return an array of same shape and dtype as a given array."""
    return allocate(a.shape, a.dtype, b, description)


def get(shape, dtype, description):
    """
    Get an array of given shape and dtype from the stack or a scratch array.
    The output array is guaranteed to be C-contiguous.
    """
    global scratch
    requested = np.product(shape) * dtype.itemsize
    buf = stack[0]
    if not buf.flags.contiguous or requested > buf.nbytes:
        buf = scratch
    else:
        buf = buf.ravel().view(np.int8, utils.ndarraywrap)

    buf, new = allocate(shape, dtype, buf, description)
    if new:
        scratch = buf.ravel().view(np.int8)
    return buf


def push(array):
    """Put an array on top of the stack."""
    stack.append(array)


def pop():
    """Remove and return the array on top of the stack."""
    return stack.pop()


def wrap_ndarray(array):
    """
    Make an input ndarray an instance of a heap class so that we can
    change its class and attributes.
    """
    if type(array) is np.ndarray:
        array = array.view(utils.ndarraywrap)
    return array
