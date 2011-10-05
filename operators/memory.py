"""
This module handles the allocation of memory.

The stack is by construction a list of contiguous int8 vectors.
"""
from __future__ import division

import gc
import numpy as np
from . import utils

__all__ = []

verbose = True
istack = 0
stack = []


def allocate(shape, dtype, buf, description):
    """
    Return an array of given shape and dtype. If a buffer is provided and
    is large enough, it is reused, otherwise a memory allocation takes
    place. Every allocation should go through this method.
    """

    if utils.isscalar(shape):
        shape = (shape,)
    dtype = np.dtype(dtype)

    requested = dtype.itemsize * reduce(lambda x, y: x * y, shape, 1)
    if buf is not None and buf.nbytes >= requested:
        if utils.isscalar(buf):
            buf = buf.reshape(1)
        buf = buf.ravel().view(np.int8)[:requested].view(dtype).reshape(shape)
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


def down():
    """
    Move stack pointer to the bottom.
    """
    global istack
    if istack == 0:
        raise ValueError('The stack pointer already is at the bottom.')
    istack -= 1


def up():
    """
    Move stack pointer to the top.
    """
    global stack, istack
    assert istack <= len(stack)
    if istack == len(stack):
        stack.append(None)
    istack += 1


def swap():
    """
    Swap stack elements istack and istack+1.
    """
    global stack, istack
    if istack == len(stack):
        stack.append(None)
    if istack == len(stack) - 1:
        stack.append(None)
    stack[istack], stack[istack + 1] = stack[istack + 1], stack[istack]


def get(nbytes, shape, dtype, description):
    """
    Get an array of given shape and dtype from the stack.
    The output array is guaranteed to be C-contiguous.
    """
    global stack, istack
    assert istack <= len(stack)
    if istack == len(stack):
        stack.append(None)

    v = stack[istack]
    if v is not None and v.nbytes >= nbytes:
        return v[:nbytes]

    if verbose:
        if nbytes < 1024:
            snbytes = str(nbytes) + ' bytes'
        else:
            snbytes = str(nbytes / 2**20) + ' MiB'
        print(
            'Info: Allocating '
            + utils.strshape(shape)
            + ' '
            + dtype.type.__name__
            + ' = '
            + snbytes
            + ' in '
            + description
            + '.'
        )

    v = None
    try:
        v = np.empty(nbytes, dtype=np.int8).view(utils.ndarraywrap)
    except MemoryError:
        gc.collect()
        v = np.empty(nbytes, dtype=np.int8).view(utils.ndarraywrap)

    stack[istack] = v
    return v


def wrap_ndarray(array):
    """
    Make an input ndarray an instance of a heap class so that we can
    change its class and attributes.
    """
    if type(array) is np.ndarray:
        array = array.view(utils.ndarraywrap)
    return array


def manager(array):
    """
    Context manager. On entering, put the input array on top of the stack,
    if it is contiguous. and pop it on exiting.
    """
    global stack, istack

    class MemoryManager(object):
        def __enter__(self):
            global stack, istack
            if array.flags.contiguous:
                stack.insert(0, array.ravel().view(np.int8, utils.ndarraywrap))
                istack = 0
            else:
                stack.insert(0, None)
                istack = 1

        def __exit__(self, *excinfo):
            stack.pop(0)

    return MemoryManager()
