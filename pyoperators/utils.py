from __future__ import division

import multiprocessing
import numpy as np
import os
import scipy.sparse


class ndarraywrap(np.ndarray):
    pass


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
    """Test if an attribute is a class attribute."""
    for c in cls.__mro__:
        if a in c.__dict__:
            return True
    return False


def isscalar(data):
    """Hack around np.isscalar oddity"""
    if isinstance(data, (list, tuple, scipy.sparse.base.spmatrix)):
        return False
    if isinstance(data, np.ndarray):
        return data.ndim == 0
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
    choices = ["'" + choice + "'" for choice in choices]
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
    if len(shape) == 1:
        return str(shape[0])
    return str(shape).replace(' ', '')


def tointtuple(data):
    """Return input as a tuple of int."""
    if data is None:
        return data
    try:
        return tuple(None if d is None else int(d) for d in data)
    except:
        return (int(data),)
