from __future__ import division

import collections
import multiprocessing
import numpy as np
import operator
import os
import scipy.sparse
import types

from . import cythonutils as cu

__all__ = ['all_eq',
           'find',
           'first_is_not',
           'inspect_special_values',
           'isclassattr',
           'isscalar',
           'merge_none',
           'ndarraywrap',
           'openmp_num_threads',
           'operation_assignment',
           'operation_symbol',
           'product',
           'strelapsed',
           'strenum',
           'strinfo',
           'strnbytes',
           'strplural',
           'strshape',
           'tointtuple']

def all_eq(a, b):
    """
    Return True if a and b are equal by recursively comparing them.
    """
    if a is b:
        return True
    if isinstance(a, collections.Mapping):
        if type(a) is not type(b):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a:
            if not all_eq(a[k], b[k]):
                return False
        return True
    if isinstance(a, (str, unicode)):
        if type(a) is not type(b):
            return False
        return a == b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(a, b)
    if isinstance(a, collections.Container):
        if type(a) is not type(b):
            return False
        if len(a) != len(b):
            return False
        for a_, b_ in zip(a, b):
            if not all_eq(a_, b_):
                return False
            return True
    if isinstance(a, types.MethodType):
        if type(a) is not type(b):
            return False
        return a.im_class is b.im_class and a.im_func is b.im_func
    if isinstance(a, types.LambdaType):
        if type(a) is not type(b):
            return False
        return a.func_code is b.func_code
    return a == b

def find(l, f):
    """
    Return first item in list that verifies a certain condition, and None
    otherwise.

    Parameters
    ----------
    l : list
        List of elements to be searched for.
    f : function
        Function that evaluates to True to match an element.

    Example:
    --------
    >>> find([1,2,3], lambda x: x > 1.5)
    2

    """
    return next((_ for _ in l if f(_)), None)

def first_is_not(l, v):
    """
    Return first item in list which is not the specified value.
    If all items are the specified value, return it.
    """
    for a in l:
        if a is not v:
            return a
    return v

def inspect_special_values(x):
    """
    If an array has no other values than -1, 0 and 1, return a tuple consisting
    of their occurences plus the boolean False and a boolean indicating if
    all values are equal. Otherwise, return the tuple (0, 0, 0, True,
    np.all(x == x.flat[0]))

    Parameter
    ---------
    x : numerical ndarray
        The array to be inspected.

    Examples
    --------
    >>> inspect_special_values([0,-1,-1])
    2, 1, 0, False, False
    >>> inspect_special_values([0,-1,-1,1.2])
    0, 0, 0, True, False

    """
    x = np.asarray(x)
    if x.size == 0:
        return 0, 0, 0, 0, False
    x = x.ravel()
    kind = x.dtype.kind
    if kind == 'b':
        return cu.inspect_special_values_bool8(x.view(np.uint8))
    if kind == 'f':
        return cu.inspect_special_values_float64(x.astype(np.float64))
    if kind == 'i':
        return cu.inspect_special_values_int64(x.astype(np.int64))
    if kind == 'u':
        return cu.inspect_special_values_uint64(x.astype(np.uint64))
    if kind == 'c':
        return cu.inspect_special_values_complex128(x.astype(np.complex128))
    return 0, 0, 0, True, False

def isclassattr(cls, a):
    """ Test if an attribute is a class attribute. """
    for c in cls.__mro__:
        if a in c.__dict__:
            return True
    return False

def isscalar(data):
    """Hack around np.isscalar oddity"""
    if isinstance(data, np.ndarray):
        return data.ndim == 0
    if isinstance(data, (str, unicode)):
        return True
    if isinstance(data, (collections.Container, scipy.sparse.base.spmatrix)):
        return False
    return True

def merge_none(a, b):
    """
    Compare two sequences elementwise and merge them discarding None entries.

    Raises ValueError exception if the two sequances do not have the same
    length or if they have different non-None elements.

    Parameters
    ----------
    a, b : sequences
        The sequences to be compared.

    Example
    -------
    >>> merge_none([1,None,3],[None,2,3])
    [1, 2, 3]
    """
    if a is b is None:
        return None
    if len(a) != len(b):
        raise ValueError('The input sequences do not have the same length.')
    if any(p != q for p,q in zip(a,b) if None not in (p,q)):
        raise ValueError('The input sequences have incompatible values.')
    return tuple(p if p is not None else q for p,q in zip(a,b))
    
class ndarraywrap(np.ndarray):
    pass

def openmp_num_threads():
    n = os.getenv('OMP_NUM_THREADS')
    if n is not None:
        return int(n)
    return multiprocessing.cpu_count()
    
def operation_assignment(a, b):
    """
    operation_assignment(a, b) -- Same as a[...] = b.
    """
    a[...] = b

operation_symbol = {
    operator.iadd: '+',
    operator.isub: '-',
    operator.imul: '*',
    operator.idiv: '/',
}

def product(a):
    """ Return the product of a arbitrary input, including generators. """
    if isinstance(a, (list, tuple, types.GeneratorType)):
        # a for loop is a bit faster than reduce(operator.imul, a)
        r = 1
        for x in a:
            r *= x
        return r

    a = np.asarray(a)
    return np.product(a, dtype=a.dtype)

def strelapsed(t0, msg='Elapsed time'):
    """
    Return an information message including elapsed time.

    Parameters
    ----------
    t0 : float
        The starting time stamp, obtained with time.time()
    msg : string, optional
        Informative message

    Example
    -------
    >>> import time
    >>> t0 = time.time()
    >>> pass
    >>> print(strelapsed(t0, 'Did nothing in'))
    Info computernode: Did nothing in... 0.00s
 
    """
    import time
    return strinfo(msg + '... {0:.2f}s'.format(time.time()-t0))[:-1]

def strenum(choices, last='or'):
    """
    Enumerates elements of a list

    Parameters
    ----------
    choices : list of string
        list of elements to be enumerated
    last : string
        last separator

    Examples
    --------
    >>> strenum(['blue', 'red', 'yellow'], 'or')
    "'blue', 'red' or 'yellow'"

    """
    choices = [ "'{0}'".format(choice) for choice in choices ]
    if len(choices) == 0:
        raise ValueError('There is no valid choice.')
    if len(choices) == 1:
        return choices[0]
    return ', '.join(choices[0:-1]) + ' ' + last + ' ' + choices[-1]

def strinfo(msg):
    """
    Return information message adding processor's node name.

    Parameter
    ---------
    msg : string
        The information message.
    Example
    -------
    >>> print(strinfo('My information message'))
    Info computernode: My information message. 

    """
    import platform
    return 'Info {0}: {1}.'.format(platform.node(), msg)

def strnbytes(nbytes):
    """
    Return number of bytes in a human readable unit of KiB, MiB or GiB.

    Parameter
    ---------
    nbytes: int
        Number of bytes, to be displayed in a human readable way.

    Example
    -------
    >>> a = np.empty((100,100))
    >>> print(strnbytes(a.nbytes))
    78.125 KiB

    """
    if nbytes < 1024:
        return str(nbytes) + ' bytes'
    elif nbytes < 1048576:
        return str(nbytes / 2**10) + ' KiB'
    elif nbytes < 1073741824:
        return str(nbytes / 2**20) + ' MiB'
    else:
        return str(nbytes / 2**30) + ' GiB'

def strplural(name, n, prepend=True, s=''):
    """
    Returns the plural or singular of a string

    Parameters
    ----------
    name : string
        string for which a plural is requested
    n : integer
        the plural or singular is based on this number
    prepend : boolean
        if true, prepend n
    s : string
        string to be appended if n > 0

    Examples
    --------
    >>> strplural('cat', 0)
    'no cat'
    >>> strplural('cat', 1)
    '1 cat'
    >>> strplural('cat', 2)
    '2 cats'
    >>> strplural('cat', 2, prepend=False)
    'cats'
    >>> animals = ['cat', 'dog']
    >>> strplural('animal', len(animals), s=': ') + ', '.join(animals)
    '2 animals: cat, dog'

    """
    if n == 0:
        return ('no ' if prepend else '') + name
    elif n == 1:
        return ('1 ' if prepend else '') + name + s
    else:
        return (str(n) + ' ' if prepend else '') + name + 's' + s

def strshape(shape):
    """ Helper function to convert shapes or list of shapes into strings. """
    if shape is None or len(shape) == 0:
        return str(shape)
    if isinstance(shape[0], tuple):
        return ', '.join(strshape(s) for s in shape)
    if len(shape) == 1:
        return str(shape[0])
    return str(shape).replace(' ','')


def tointtuple(data):
    """Return input as a tuple of int."""
    if data is None:
        return data
    try:
        return tuple(None if d is None else int(d) for d in data)
    except TypeError:
        return (int(data),)
