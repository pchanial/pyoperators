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

    requested = dtype.itemsize * reduce(lambda x,y:x*y, shape, 1)
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
        print('Info: Allocating ' + str(shape).replace(' ','') + ' ' + \
              dtype.type.__name__ + ' = ' + snbytes + ' in ' + \
              description + '.')
    try:
        buf = np.empty(shape, dtype)
    except MemoryError:
        gc.collect()
        buf = np.empty(shape, dtype)

    return wrap_ndarray(buf), True

def allocate_like(a, b, description):
    """ Return an array of same shape and dtype as a given array. """
    return allocate(a.shape, a.dtype, b, description)

def down():
    global istack
    istack -= 1

def up():
    global stack, istack
    assert istack <= len(stack)
    if istack == len(stack):
        stack.append(None)
    istack += 1

def get(shape, dtype, description):
    """
    Get an array of given shape and dtype from the stack.
    The output array is guaranteed to be C-contiguous.
    """
    global stack, istack
    if istack == 0:
        requested = dtype.itemsize * reduce(lambda x,y:x*y, shape, 1)
        if requested > stack[0].size:
            istack += 1
    
    assert istack <= len(stack)

    if istack == len(stack):
        stack.append(None)

    buf, new = allocate(shape, dtype, stack[istack], description)
    if new:
        stack[istack] = buf.ravel().view(np.int8)

    return buf

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
            if array.flags.contiguous and array.ndim > 0:
                stack.insert(0, array.view(np.int8, utils.ndarraywrap).ravel())
                istack = 0
            else:
                stack.insert(0, None)
                istack = 1
        def __exit__(self, *excinfo):
            stack.pop(0)
    return MemoryManager()

