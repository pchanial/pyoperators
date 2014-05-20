from __future__ import absolute_import, division, print_function

import collections
import itertools
import multiprocessing
import numpy as np
import operator
import os
import signal
import types

from contextlib import contextmanager
from itertools import izip
from . import cythonutils as cu
from ..warnings import warn, PyOperatorsDeprecationWarning

__all__ = [
    'all_eq',
    'broadcast_shapes',
    'cast',
    'complex_dtype',
    'first',
    'first_is_not',
    'float_dtype',
    'groupbykey',
    'ifirst',
    'ifirst_is_not',
    'inspect_special_values',
    'interruptible',
    'interruptible_if',
    'isalias',
    'isclassattr',
    'isscalar',
    'isscalarlike',
    'izip_broadcast',
    'least_greater_multiple',
    'merge_none',
    'ndarraywrap',
    'one',
    'openmp_num_threads',
    'operation_assignment',
    'operation_symbol',
    'pi',
    'product',
    'renumerate',
    'reshape_broadcast',
    'setting',
    'strelapsed',
    'strenum',
    'strinfo',
    'strnbytes',
    'strplural',
    'strshape',
    'tointtuple',
    'uninterruptible',
    'uninterruptible_if',
    'zero',
]


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
    if isinstance(a, (float, np.ndarray, np.number)) or isinstance(
        b, (float, np.ndarray, np.number)
    ):
        return np.allclose(a, b)
    if isinstance(a, collections.Container):
        if type(a) is not type(b):
            return False
        if len(a) != len(b):
            return False
        for a_, b_ in izip(a, b):
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


def broadcast_shapes(*shapes):
    """
    Broadcast any number of shapes against each other.

    Parameters
    ----------
    *shapes : tuples
        The shapes to broadcast

    Example
    -------
    >>> broadcast_shapes((1,5), (3, 2, 1))
    (3, 2, 5)

    """
    if any(not isinstance(s, tuple) for s in shapes):
        raise TypeError('The input shapes are not tuples.')
    ndim = max(len(s) for s in shapes)
    shapes_ = [(ndim - len(s)) * [1] + list(s) for s in shapes]
    outshape = []
    for idim, dims in enumerate(zip(*shapes_)):
        dims = [dim for dim in dims if dim != 1]
        if len(dims) == 0:
            d = 1
        elif any(dim != dims[0] for dim in dims):
            raise ValueError(
                'The shapes could not be broadcast together {}'.format(
                    ' '.join(str(s) for s in shapes)
                )
            )
        else:
            d = dims[0]
        outshape.append(d)
    return tuple(outshape)


def cast(arrays, dtype=None, order='c'):
    """
    Cast a list of arrays into a same data type.

    Parameters
    ----------
    arrays : sequence of array-like or None
        The list of arrays to be cast.
    dtype : numpy.dtype
        If specified, all arrays will be cast to this data type. Otherwise,
        the data types is inferred from the arrays.

    Example
    -------
    >>> cast([[1., 2.], None, np.array(2j)])
    (array([ 1.+0.j,  2.+0.j]), None, array(2j))

    """
    arrays = tuple(arrays)
    if dtype is None:
        arrays_ = [np.array(a, copy=False) for a in arrays if a is not None]
        dtype = np.result_type(*arrays_)
    result = (
        np.array(a, dtype=dtype, order=order, copy=False) if a is not None else None
        for a in arrays
    )
    return tuple(result)


def complex_dtype(dtype):
    """
    Return the complex dtype associated to a numeric dtype.

    Parameter
    ---------
    dtype : dtype
        The input dtype.

    Example
    -------
    >>> complex_dtype(int)
    dtype('complex128')
    >>> complex_dtype(np.float32)
    dtype('complex64')
    >>> complex_dtype(np.float64)
    dtype('complex128')

    """
    dtype = float_dtype(dtype)
    if dtype.kind == 'c':
        return dtype
    if dtype == np.float16:
        if not hasattr(np, 'complex32'):
            return np.dtype(complex)
    return np.dtype('complex{}'.format(2 * int(dtype.name[5:])))


def float_dtype(dtype):
    """
    Return the floating dtype associated to a numeric dtype.
    Unless the input dtype kind is float or complex, the default float dtype
    is returned.

    Parameter
    ---------
    dtype : dtype
        The input dtype.

    Example
    -------
    >>> float_dtype(int)
    dtype('float64')
    >>> float_dtype(np.float32)
    dtype('float32')
    >>> float_dtype(np.complex256)
    dtype('complex256')

    """
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biufc':
        raise TypeError('Non numerical data type.')
    if dtype.kind in 'iub':
        return np.dtype(float)
    return dtype


def first(l, f, reverse=False):
    """
    Return first item in list that verifies a certain condition, or raise
    a ValueError exception otherwise.

    Parameters
    ----------
    l : list
        List of elements to be searched for.
    f : function
        Function that evaluates to True to match an element.
    reverse : boolean
        True for reverse search.

    Example:
    --------
    >>> first([1.,2.,3.], lambda x: x > 1.5)
    2.0
    >>> first([1.,2.,3.], lambda x: x > 1.5, reverse=True)
    3.0

    """
    if reverse:
        return first(reversed(tuple(l)), f)

    try:
        return next((_ for _ in l if f(_)))
    except StopIteration:
        raise ValueError('There is no matching item in the list.')


def first_is_not(l, v, reverse=False):
    """
    Return first item in list which is not the specified value.
    If all items are the specified value, return it.

    Parameters
    ----------
    l : sequence
        The list of elements to be inspected.
    v : object
        The value not to be matched.
    reverse : boolean
        True for reverse search.

    Example:
    --------
    >>> first_is_not(['a', 'b', 'c'], 'a')
    'b'
    >>> first_is_not(['a', 'b', 'c'], 'b', reverse=True)
    'c'

    """
    if reverse:
        return first_is_not(reversed(tuple(l)), v)
    return next((_ for _ in l if _ is not v), v)


def groupbykey(iterable, key):
    """
    Create an iterator which returns (key, sub-iterator) grouped by each
    value of key.

    """
    iterator = izip(iterable, key)
    i, value = next(iterator)
    l = [i]
    for i, k in iterator:
        if k == value:
            l.append(i)
            continue
        yield value, l
        value = k
        l = [i]
    if len(l) != 0:
        yield value, l


def ifirst(l, match, reverse=False):
    """
    Return the index of the first item in a list that verifies a certain
    condition or is equal to a certain value. Raise a ValueError exception
    otherwise.

    Parameters
    ----------
    l : iterator
        List of elements to be searched for.
    match : callable or object
        Function that evaluates to True to match an element or the element
        to be matched.
    reverse : boolean
        True for reverse search.

    Example:
    --------
    >>> ifirst([1.,2.,3.], lambda x: x > 1.5)
    1
    >>> ifirst([1.,2.,3.], lambda x: x > 1.5, reverse=True)
    2
    >>> ifirst([1., 2., 3.], 2)
    1

    """
    if reverse:
        l = tuple(l)
        index = ifirst(reversed(l), match)
        return len(l) - index - 1

    try:
        if not callable(match):
            return next((i for i, _ in enumerate(l) if _ == match))
        return next((i for i, _ in enumerate(l) if match(_)))
    except StopIteration:
        raise ValueError('There is no matching item in the list.')


def ifirst_is_not(l, v, reverse=False):
    """
    Return index of first item in list which is not the specified value.
    If the list is empty or if all items are the specified value, raise
    a ValueError exception.

    Parameters
    ----------
    l : sequence
        The list of elements to be inspected.
    v : object
        The value not to be matched.
    reverse : boolean
        True for reverse search.

    Example:
    --------
    >>> ifirst_is_not(['a', 'b', 'c'], 'a')
    1
    >>> ifirst_is_not(['a', 'b', 'c'], 'a', reverse=True)
    2

    """
    if reverse:
        l = tuple(l)
        index = ifirst_is_not(reversed(l), v)
        return len(l) - index - 1
    try:
        return next((i for i, _ in enumerate(l) if _ is not v))
    except StopIteration:
        raise ValueError('There is no matching item in the list.')


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


@contextmanager
def interruptible():
    """Make a block of code interruptible with CTRL-C."""
    signal_old = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, signal_old)


@contextmanager
def interruptible_if(condition):
    """Conditionally make a block of code interruptible with CTRL-C."""
    if not condition:
        yield
    else:
        with interruptible():
            yield


def isalias(array1, array2):
    """
    Return True if the two input arrays point to the same memory location.

    """
    return (
        array1.__array_interface__['data'][0] == array2.__array_interface__['data'][0]
    )


def isclassattr(cls, a):
    """Test if an attribute is a class attribute."""
    for c in cls.__mro__:
        if a in c.__dict__:
            return True
    return False


def isscalar(x):
    """Deprecated."""
    warn('Use isscalarlike instead of isscalar.', PyOperatorsDeprecationWarning)
    return isscalarlike(x)


def isscalarlike(x):
    """Return True for scalars and 0-ranked arrays."""
    return np.isscalar(x) or isinstance(x, np.ndarray) and x.ndim == 0


def izip_broadcast(*args):
    """
    Like izip, except that arguments which are containers of length 1 are
    repeated.

    """

    def wrap(a):
        if hasattr(a, '__len__') and len(a) == 1:
            return itertools.repeat(a[0])
        return a

    if any(not hasattr(a, '__len__') or len(a) != 1 for a in args):
        args = [wrap(arg) for arg in args]
    return izip(*args)


def least_greater_multiple(a, l, out=None):
    """
    Return the least multiple of values in a list greater than a given number.

    Example
    -------
    >>> least_greater_multiple(2253, [2,3])
    2304

    """
    if any(v <= 0 for v in l):
        raise ValueError('The list of multiple is not positive;')
    it = np.nditer(
        [a, out], op_flags=[['readonly'], ['writeonly', 'allocate', 'no_broadcast']]
    )
    max_power = [int(np.ceil(np.log(np.max(a)) / np.log(v))) for v in l]
    slices = [slice(0, m + 1) for m in max_power]
    powers = np.ogrid[slices]
    values = 1
    for v, p in izip(l, powers):
        values = values * v**p
    for v, o in it:
        if np.__version__ < '2':
            values_ = np.ma.MaskedArray(values, mask=values < v, copy=False)
            o[...] = np.min(values_)
        else:
            o[...] = np.amin(values, where=values >= v)
    out = it.operands[1]
    if out.ndim == 0:
        return out.flat[0]
    return out


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
    if any(p != q for p, q in izip(a, b) if None not in (p, q)):
        raise ValueError('The input sequences have incompatible values.')
    return tuple(p if p is not None else q for p, q in izip(a, b))


class ndarraywrap(np.ndarray):
    pass


def one(dtype):
    """Return 1 with a given dtype."""
    return np.ones((), dtype=dtype)[()]


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


def pi(dtype):
    """Return pi with a given dtype."""
    return 4 * np.arctan(one(dtype))


def product(a):
    """Return the product of a arbitrary input, including generators."""
    if isinstance(a, (list, tuple, types.GeneratorType)):
        # a for loop is a bit faster than reduce(operator.imul, a)
        r = 1
        for x in a:
            r *= x
        return r

    a = np.asarray(a)
    return np.product(a, dtype=a.dtype)


def renumerate(l):
    """Reversed enumerate."""
    return izip(xrange(len(l) - 1, -1, -1), reversed(l))


def reshape_broadcast(x, shape):
    """
    Reshape an array by setting broadcastable dimensions' strides to zero.

    Parameters
    ----------
    x : array-like
        The array to be reshaped.
    shape : tuple of int
        New shape of array. It can be any positive number along the axes of x
        of length 1.

    Example
    -------
    >>> a = np.arange(3).reshape((3, 1))
    >>> b = reshape_broadcast(a, (2, 3, 2))
    >>> print(b)
    [[[0 0]
      [1 1]
      [2 2]]

     [[0 0]
      [1 1]
      [2 2]]]
    >>> b.shape
    (2, 3, 2)
    >>> b.strides
    (0, 8, 0)

    """
    x = np.asanyarray(x)
    if len(shape) < x.ndim or any(
        os != 1 and os != ns for os, ns in zip(x.shape, shape[-x.ndim :])
    ):
        raise ValueError(
            "The requested shape '{0}' is incompatible with that "
            "of the array '{1}'.".format(shape, x.shape)
        )
    strides = (len(shape) - x.ndim) * (0,) + tuple(
        (0 if sh == 1 else st for sh, st in zip(x.shape, x.strides))
    )
    return np.lib.stride_tricks.as_strided(x, shape, strides)


@contextmanager
def setting(obj, attr, value):
    """Contextually set an attribute to an object."""
    if hasattr(obj, attr):
        old_value = getattr(obj, attr)
        do_delete = False
    else:
        do_delete = True
    setattr(obj, attr, value)
    yield
    if do_delete:
        delattr(obj, attr)
    else:
        setattr(obj, attr, old_value)


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

    return strinfo(msg + '... {0:.2f}s'.format(time.time() - t0))[:-1]


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
    >>> strenum(['blue', 'red', 'yellow'])
    "'blue', 'red' or 'yellow'"

    """
    choices = ["'{0}'".format(choice) for choice in choices]
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


def strplural(n, name, nonumber=False, s=''):
    """
    Returns the plural or singular of a string

    Parameters
    ----------
    n : integer
        The plural or singular is based on this number.
    name : string
        String for which a plural is requested.
    nonumber : boolean
        If true, don't prepend the number.
    s : string
        String to be appended if n > 0

    Examples
    --------
    >>> strplural(0, 'cat')
    'no cat'
    >>> strplural(1, 'cat')
    '1 cat'
    >>> strplural(2, 'cat')
    '2 cats'
    >>> strplural(2, 'cat', prepend=False)
    'cats'
    >>> animals = ['cat', 'dog']
    >>> strplural(len(animals), 'animal', s=': ') + ', '.join(animals)
    '2 animals: cat, dog'
    >>> strplural(0, 'animal', s=':')
    'no animal'

    """
    if n == 0:
        return ('' if nonumber else 'no ') + name
    elif n == 1:
        return ('' if nonumber else '1 ') + name + s
    else:
        return ('' if nonumber else str(n) + ' ') + name + 's' + s


def strshape(shape, broadcast=None):
    """Helper function to convert shapes or list of shapes into strings."""
    if shape is None:
        return str(shape)
    if not isinstance(shape, tuple):
        raise TypeError('Invalid shape.')
    if len(shape) == 0 and broadcast in ('leftward', 'rightward'):
        return '(...)'
    if broadcast == 'leftward':
        shape = ('...',) + shape
    elif broadcast == 'rightward':
        shape = shape + ('...',)
    if len(shape) == 0:
        return str(shape)
    if len(shape) == 1:
        return str(shape[0])
    return str(shape).replace(' ', '').replace("'", '')


def tointtuple(data):
    """Return input as a tuple of int."""
    if data is None:
        return data
    try:
        return tuple(None if d is None else int(d) for d in data)
    except TypeError:
        return (int(data),)


@contextmanager
def uninterruptible():
    """
    Make a block of code uninterruptible with CTRL-C.
    The KeyboardInterrupt is re-raised after the block is executed.

    """
    signal_old = signal.getsignal(signal.SIGINT)
    # XXX the nonlocal Python3 would be handy here
    ctrlc_is_pressed = []

    def signal_handler(signal, frame):
        ctrlc_is_pressed.append(True)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        yield
    except:
        raise
    finally:
        signal.signal(signal.SIGINT, signal_old)
        if len(ctrlc_is_pressed) > 0:
            raise KeyboardInterrupt()


@contextmanager
def uninterruptible_if(condition):
    """Conditionally make a block of code uninterruptible with CTRL-C."""
    if not condition:
        yield
    else:
        with uninterruptible():
            yield


def zero(dtype):
    """Return 0 with a given dtype."""
    return np.zeros((), dtype=dtype)[()]
