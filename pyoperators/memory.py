"""
This module handles the allocation of memory.

The stack is by construction a list of contiguous int8 vectors.
In addition to temporary arrays that are used for intermediate operations,
the stack may contain the array that will be the output of the operator.
Care has been taken to ensure that the latter is released from the stack
to avoid side effects.
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
        buf = buf.ravel().view(np.int8)[:requested].view(dtype).reshape(shape)
        return buf, False

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

    return buf, True

def allocate_like(a, b, description):
    """ Return an array of same shape and dtype as a given array. """
    return allocate(a.shape, a.dtype, b, description)

def clear():
    """ Clear the memory stack. """
    global stack, istack
    stack = []
    istack = 0

def down():
    """
    Move stack pointer towards the bottom.
    """
    global istack
    if istack == 0:
        raise ValueError('The stack pointer already is at the bottom.')
    istack -= 1

def up():
    """
    Move stack pointer towards the top.
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
    stack[istack], stack[istack+1] = stack[istack+1], stack[istack]

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
        print('Info: Allocating ' + utils.strshape(shape) + ' ' + \
              dtype.type.__name__ + ' = ' + snbytes + ' in ' + \
              description + '.')

    v = None
    try:
        v = np.empty(nbytes, dtype=np.int8)
    except MemoryError:
        gc.collect()
        v = np.empty(nbytes, dtype=np.int8)
    
    stack[istack] = v
    return v

def push_and_pop(array):
    """
    Return a context manager.
    If the input array is contiguous, push it on top of the stack on entering
    and pop it on exiting.
    """
    class MemoryManager(object):
        def __enter__(self):
            global stack, istack
            if array.flags.contiguous:
                array_ = array.ravel().view(np.int8)
                self.id = id(array_)
                stack.insert(istack, array_)
            else:
                istack += 1
        def __exit__(self, *excinfo):
            global stack, istack
            if array.flags.contiguous:
                assert self.id == id(stack[istack])
                stack.pop(istack)
            else:
                istack -= 1
    return MemoryManager()

