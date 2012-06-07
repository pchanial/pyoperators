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
from .utils import product

__all__ = []

verbose = False
istack = 0
stack = []

def allocate(shape, dtype, description, verbose=None):
    """
    Return an array of given shape and dtype.

    """
    if utils.isscalar(shape):
        shape = (shape,)
    dtype = np.dtype(dtype)
    if verbose is None:
        verbose = globals()['verbose']

    if verbose:
        requested = dtype.itemsize * reduce(lambda x,y:x*y, shape, 1)
        if requested > 0:
            print(utils.strinfo('Allocating ' + str(shape).replace(' ','') +
                  ' ' + (str(dtype) if dtype.kind != 'V' else 'elements') +
                  ' = ' + utils.strnbytes(requested) + ' ' + description))
    try:
        buf = np.empty(shape, dtype)
    except MemoryError:
        gc.collect()
        buf = np.empty(shape, dtype)

    return buf

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

def get(shape, dtype, description):
    """
    Get an array of given shape and dtype from the stack.
    The output array is guaranteed to be C-contiguous.
    """
    global stack, istack    
    assert istack <= len(stack)

    dtype = np.dtype(dtype)
    nbytes = product(shape) * dtype.itemsize

    if istack == len(stack):
        stack.append(None)

    v = stack[istack]
    if v is not None and v.nbytes >= nbytes:
        return v[:nbytes].view(dtype).reshape(shape)

    if verbose:
        print(utils.strinfo('Allocating ' + utils.strshape(shape) + ' ' +
              dtype.type.__name__ + ' = ' + utils.strnbytes(nbytes) + ' in ' +
              description))

    v = None
    try:
        v = np.empty(nbytes, dtype=np.int8)
    except MemoryError:
        gc.collect()
        v = np.empty(nbytes, dtype=np.int8)
    
    stack[istack] = v
    return v.view(dtype).reshape(shape)

def print_stack(stack_addresses={}):
    """
    Print the stack.
    A dict of ndarray addresses can be used to name the stack elements.

    Example
    -------
    print_stack({'output':myoutput})

    """
    global stack, istack
    if len(stack) == 0:
        print 'The memory stack is empty.'
    d = dict((v.__array_interface__['data'][0] if isinstance(v, np.ndarray)
             else v, k) for k, v in stack_addresses.items())
    result = []
    for i, s in enumerate(stack):
        pointer = '--> ' if i == istack else '    '
        res = pointer + '{0:<2}: '.format(i)
        if s is None:
            res += 'None'
        else:
            address = s.__array_interface__['data'][0]
            if address in d:
                strid = d[address] + ' '
            else:
                strid = ''
            strid += hex(address)
            res += '{1}\t({2} bytes)'.format(i, strid, s.nbytes)
        result.insert(0, res)
    print '\n'.join(result)
    
def push_and_pop(array):
    """
    Return a context manager.
    If the input array is contiguous, push it on top of the stack on entering
    and pop it on exiting.
    """
    from .utils.mpi import MPI

    class MemoryManager(object):

        def __enter__(self):
            global stack, istack
            self.istack = istack
            if array.flags.contiguous:
                array_ = array.ravel().view(np.int8)
                self.address = array_.__array_interface__['data'][0]
                stack.insert(istack, array_)

        def __exit__(self, *excinfo):
            import sys
            global stack, istack
            if sys.exc_info()[0] is None:
                assert istack == self.istack
                if not array.flags.contiguous:
                    return
                address = stack[istack].__array_interface__['data'][0]
                if self.address != address:
                    msg = 'Stack'
                    if MPI.COMM_WORLD.size > 1:
                        msg += ' for rank ' + str(MPI.COMM_WORLD.rank)
                    print(msg + ', failed to find ' + hex(self.address) +
                          ' at istack=' + str(istack) + ':')
                    print_stack({self.address : 'pushed array'})
                    stack.pop(istack)
                    assert False
            else:
                istack = self.istack
                if not array.flags.contiguous:
                    return
            stack.pop(istack)

    return MemoryManager()

