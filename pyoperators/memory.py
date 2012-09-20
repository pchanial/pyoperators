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
import inspect
import numpy as np
from contextlib import contextmanager
from . import utils
from .utils import ifind, product, strshape, tointtuple

__all__ = ['empty', 'ones', 'zeros']

MEMORY_ALIGNMENT = 32
MEMORY_TOLERANCE = 1.2 # We allow reuse of pool variables only if they do not
                       # exceed 20% of the requested size

verbose = False

def empty(shape, dtype=np.float, order='c', description=None, verbose=None):
    """
    Return a new aligned and contiguous array of given shape and type, without
    initializing entries.

    """
    shape = tointtuple(shape)
    dtype = np.dtype(dtype)
    if verbose is None:
        verbose = globals()['verbose']

    requested = product(shape) * dtype.itemsize
    if verbose and requested > 0:
        if description is None:
            frames = inspect.getouterframes(inspect.currentframe())
            i = 1
            while True:
                if frames[i][1].replace('.pyc', '.py') != \
                   __file__.replace('.pyc', '.py'):
                    break
                i += 1
            description = frames[i][3].replace('<module>', 'session')
            if 'self' in frames[i][0].f_locals:
                cls = type(frames[i][0].f_locals['self']).__name__
                description = cls + '.' + description
            description = 'in ' + description
        print(utils.strinfo('Allocating ' + strshape(shape) +
              ' ' + (str(dtype) if dtype.kind != 'V' else 'elements') +
              ' = ' + utils.strnbytes(requested) + ' ' + description))
    try:
        buf = np.empty(requested + MEMORY_ALIGNMENT, np.int8)
    except MemoryError:
        gc.collect()
        buf = np.empty(requested + MEMORY_ALIGNMENT, np.int8)

    address = buf.__array_interface__['data'][0]
    offset = MEMORY_ALIGNMENT - address % MEMORY_ALIGNMENT

    return np.frombuffer(buf.data, np.int8, count=requested, offset=offset) \
             .view(dtype).reshape(shape, order=order)

def ones(shape, dtype=np.float, order='c', description=None, verbose=None):
    """
    Return a new aligned and contiguous array of given shape and type, filled
    with ones.

    """
    a = empty(shape, dtype, order, description, verbose)
    a[...] = 1
    return a

def zeros(shape, dtype=np.float, order='c', description=None, verbose=None):
    """
    Return a new aligned and contiguous array of given shape and type, filled
    with zeros.

    """
    a = empty(shape, dtype, order, description, verbose)
    a[...] = 0
    return a

def is_compatible(array, shape, dtype, alignment=1, contiguous=False,
                  tolerance=np.inf):
    """
    Return True if a buffer with specified requirements can be extracted
    from an numpy array.

    """
    shape = tointtuple(shape)
    dtype = np.dtype(dtype)
    if array.__array_interface__['data'][0] % alignment != 0:
        return False
    if not array.flags.contiguous:
        if contiguous:
            return False
        return array.shape == shape and array.itemsize == dtype.itemsize
    nbytes = product(shape) * dtype.itemsize
    return array.nbytes >= nbytes and array.nbytes / nbytes <= tolerance
    
class MemoryPool(object):
    """
    Class implementing a pool of buffers.
    """

    def __init__(self):
        self._buffers = []

    def add(self, v):
        """ Add a numpy array to the pool. """
        if not isinstance(v, np.ndarray):
            raise TypeError('The input is not an ndarray.')
        if v.flags.contiguous:
            v = v.ravel().view(np.int8)
        a = v.__array_interface__['data'][0]
        if any(_.__array_interface__['data'][0] == a for _ in self._buffers):
            raise ValueError('There already is an entry in the pool pointing to'
                             ' this memory location.')
        try:
            i = ifind(self._buffers, lambda x: x.nbytes > v.nbytes)
        except ValueError:
            i = len(self._buffers)
        self._buffers.insert(i, v)

    def clear(self):
        """
        Clear the pool.

        """
        self._buffers = []
        gc.collect()

    @contextmanager
    def get(self, shape, dtype, alignment=1, contiguous=False, description=None,
            verbose=None):
        """
        Return a context manager which retrieves a buffer from the pool
        on enter, and set it back in the pool on exit.

        """
        v_ = self.extract(shape, dtype, alignment, contiguous, description,
                          verbose)
        v = self.view(v_, shape, dtype)

        yield v
        self.add(v_)

    @contextmanager
    def get_if(self, condition, shape, dtype, description=None, verbose=None):
        """
        Return a context manager which conditionally retrieves a buffer
        from the pool on enter, and set it back in the pool on exit.

        """
        if not condition:
            yield None
        else:
            with self.get(shape, dtype, description=description,
                          verbose=verbose) as v:
                yield v

    def extract(self, shape, dtype, alignment=1, contiguous=False,
                description=None, verbose=None):
        """
        Extract a buffer from the pool given the following requirements:
        shape, dtype, alignment, contiguity.

        """
        shape = tointtuple(shape)
        dtype = np.dtype(dtype)
        compatible = lambda x: is_compatible(x, shape, dtype, alignment,
                                             contiguous, MEMORY_TOLERANCE)
        try:
            i = ifind(self._buffers, compatible)
            v = self._buffers.pop(i)
        except ValueError:
            v = empty(shape, dtype, description, verbose)
        return v

    def remove(self, v):
        """
        Remove an entry from the pool.

        """
        address = v.__array_interface__['data'][0]
        i = ifind((_.__array_interface__['data'][0] for _ in self._buffers),
                  lambda x: x == address)
        self._buffers.pop(i)

    @contextmanager
    def set(self, v):
        """
        Return a context manager that adds a buffer on enter, and remove it
        on exit.

        """
        self.add(v)
        yield
        self.remove(v)

    @contextmanager
    def set_if(self, condition, v):
        """
        Return a context manager that conditionally adds a buffer on enter,
        and remove it on exit.

        """
        if not condition:
            yield
        else:
            with self.set(v):
                yield

    @staticmethod
    def view(buf, shape, dtype):
        """
        Return a view of given shape and dtype from a buffer.
        """
        shape = tointtuple(shape)
        dtype = np.dtype(dtype)
        if buf.flags.contiguous:
            buf = buf.ravel().view(np.int8)
        elif buf.shape == shape and buf.itemsize == dtype.itemsize:
            return buf
        else:
            raise ValueError('Shape mismatch.')
        required = dtype.itemsize * product(shape)
        return buf[:required].view(dtype).reshape(shape)
        
    def __contains__(self, v):
        if not isinstance(v, np.ndarray):
            raise TypeError('The input is not an ndarray.')
        address = v.__array_interface__['data'][0]
        try:
            ifind((_.__array_interface__['data'][0] for _ in self._buffers),
                  lambda x: x == address)
        except ValueError:
            return False
        return True

    def __getitem__(self, index):
        """ Return pool entry by index. """
        return self._buffers[index]

    def __len__(self):
        """ Return the number of entries in the pool. """
        return len(self._buffers)

    def __str__(self, names={}):
        """
        Print the stack.
        A dict of ndarray addresses can be used to name the stack elements.

        Example
        -------
        print(pool.__str__({'output':myoutput}))

        """
        if len(self) == 0:
            return 'The memory stack is empty.'
        d = dict((v.__array_interface__['data'][0] if isinstance(v, np.ndarray)
                 else v, k) for k, v in names.items())
        result = []
        for i, s in enumerate(self._buffers):
            res = '{0:<2}: '.format(i)
            address = s.__array_interface__['data'][0]
            if address in d:
                strid = d[address] + ' '
            else:
                strid = ''
            strid += hex(address)
            res += '{1}\t({2} bytes)'.format(i, strid, s.nbytes)
            result.append(res)
        return '\n'.join(result)
