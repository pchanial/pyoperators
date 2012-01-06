from __future__ import division

import collections
import multiprocessing
import numpy as np
import os
import scipy.sparse

class ndarraywrap(np.ndarray):
    pass

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
    except:
        return (int(data),)

def _get_msg(msg):
    if not msg:
        return '.'
    return ': ' + str(msg) + '.'
