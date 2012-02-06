from __future__ import division

import collections
import multiprocessing
import numpy as np
import operator
import os
import scipy.sparse
import types

__all__ = [ 'operation_assignment' ]

class ndarraywrap(np.ndarray):
    pass

def all_eq(a, b):
    """
    Return True if a and b are equal by recursively comparing them.
    """
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

def assert_eq(a, b, msg=None):
    """ Assert that the two arguments are (almost) equal. """
    assert all_eq(a, b), msg

def assert_in(a, b, msg=None):
    """ Assert that the first argument is in the second one. """
    if a in b:
        return
    assert False, str(a) + ' is not in ' + str(b) + _get_msg(msg)

def assert_not_in(a, b, msg=None):
    """ Assert that the first argument is not in second one. """
    if a not in b:
        return
    assert False, str(a) + ' is in ' + str(b) + _get_msg(msg)

def assert_is(a, b, msg=None):
    """ Assert arguments are equal as determined by the 'is' operator. """
    if a is b:
        return
    assert False, str(a) + ' is not ' + str(b) + _get_msg(msg)

def assert_is_not(a, b, msg=None):
    """ Assert arguments are not equal as determined by the 'is' operator. """
    if a is not b:
        return
    assert False, str(a) + ' is ' + str(b) + _get_msg(msg)

def assert_is_instance(a, cls, msg=None):
    """ Assert that the first argument is an instance of the second one. """
    if isinstance(a, cls):
        return
    assert False, str(a) + " is not a '" + cls.__name__ + "' instance" + \
           _get_msg(msg)

def assert_is_not_instance(a, cls, msg=None):
    """ Assert that the first argument is not an instance of the second one. """
    if not isinstance(a, cls):
        return
    assert False, str(a) + " is a '" + cls.__name__ + "' instance" + \
           _get_msg(msg)

def assert_is_none(a, msg=None):
    """ Assert argument is None. """
    if a is None:
        return
    assert False, str(a) + ' is not None' + _get_msg(msg)

def assert_is_not_none(a, msg=None):
    """ Assert argument is not None. """
    if a is not None:
        return
    assert False, str(a) + ' is None' + _get_msg(msg)

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

def first_is_not(l, v):
    """
    Return first item in list which is not the specified value.
    If all items are the specified value, return it.
    """
    for a in l:
        if a is not v:
            return a
    return v

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
    if len(a) != len(b):
        raise ValueError('The input sequences do not have the same length.')
    if any(p != q for p,q in zip(a,b) if None not in (p,q)):
        raise ValueError('The input sequences have incompatible values.')
    return tuple(p if p is not None else q for p,q in zip(a,b))
    
def openmp_num_threads():
    n = os.getenv('OMP_NUM_THREADS')
    if n is not None:
        return int(n)
    return multiprocessing.cpu_count()
    
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
    choices = [ "'" + choice + "'" for choice in choices ]
    return ', '.join(choices[0:-1]) + ' ' + last + ' ' + choices[-1]

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

def _get_msg(msg):
    if not msg:
        return '.'
    return ': ' + str(msg) + '.'
