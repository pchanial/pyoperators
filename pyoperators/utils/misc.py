from __future__ import division

import collections
import gc
import multiprocessing
import numpy as np
import operator
import os
import cPickle as pickle
import scipy.sparse
import signal
import timeit
import types

from contextlib import contextmanager
from itertools import izip
from . import cythonutils as cu

__all__ = ['all_eq',
           'benchmark',
           'cast',
           'find',
           'first_is_not',
           'ifind',
           'inspect_special_values',
           'interruptible',
           'interruptible_if',
           'isalias',
           'isclassattr',
           'isscalar',
           'least_greater_multiple',
           'memory_usage',
           'merge_none',
           'ndarraywrap',
           'openmp_num_threads',
           'operation_assignment',
           'operation_symbol',
           'product',
           'renumerate',
           'strelapsed',
           'strenum',
           'strinfo',
           'strnbytes',
           'strplural',
           'strshape',
           'tointtuple',
           'uninterruptible',
           'uninterruptible_if']

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
    if isinstance(a, (float, np.ndarray, np.number)) or \
       isinstance(b, (float, np.ndarray, np.number)):
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

def benchmark(stmt, args=None, keywords=None, ids=None, setup='pass',
              filename=None):
    """
    Automate the creation of benchmark tables for functions.

    This tool benchmarks a function given a sequence of different arguments,
    keywords or identifiers. Note that a little overhead is incurred.
    It returns a tuple containing 1) an ordered dict, whose keys are formed
    from the arguments and keywords if the identifiers are not provided and
    whose values are the timings in seconds and 2) the memory usage difference
    before and after each timing.

    Parameters
    ----------
    stmt : callable or string
        The function or snippet to be timed. In case of a string, arguments and
        keywords are matched by the substrings '*args' and '*keywords'.
        Caveat: these must be interpreted correctly through their repr.
        Otherwise, they should be passed as a string (then both stmt and setup
        should be a string). See example below.
    args : sequence or iterator of sequences of arguments, optional
        The different arguments that will be passed to the function.
    keywords : sequence or iterator of dictionaries of keywords, optional
        The different keywords that will be passed to the function.
    ids : sequence or iterator of string, optional
        The identifier for each timing. If not provided, it is inferred
        from the arguments and the keywords.
    setup : callable or string, optional
        Initialisation before timing. In case of a string, arguments and
        keywords are passed with the same mechanism as the stmt argument.
    filename : string, optional
        Name of the file to save the result as a pickle file.

    Example
    -------
    >>> def f(dtype, n=10):
    ...     return np.zeros(n, dtype)
    >>> b = benchmark(f, [(float,), (int,)], [{'n':10}, {'n':100}],
    ...               ['n=10', 'n=100'])
    n=10: 100000 loops, best of 3: 2.12 usec per loop
    n=100: 100000 loops, best of 3: 2.21 usec per loop

    >>> class A():
    ...     def __init__(self, n, dtype=float):
    ...         self.a = 2 * np.ones(n, dtype)
    ...     def run(self):
    ...         np.sqrt(self.a)
    >>> b = benchmark('a.run()', ['1,dtype=int', '10,dtype=float'],
    ...               setup='from __main__ import A; a=A(*args)')

    Overhead:
    >>> def f():
    ...     pass
    >>> b = benchmark(f)
    1000000 loops, best of 3: 0.562 usec per loop

    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError('The argument stmt is neither a string nor callable.')

    if not callable(setup) and not isinstance(setup, str):
        raise TypeError('The argument setup is neither a string nor callable.')

    precision = 3
    repeat = 3

    class wrapper(object):
        def __call__(self):
            stmt(*self.args, **self.keywords)

    w = wrapper()
    single_test = args is keywords is None

    def default_args():
        while True:
            yield ()
    def default_keywords():
        while True:
            yield {}
    def default_ids():
        while True:
            id_ = '' if len(w.args) == 0 else 'args:' + str(w.args)
            if len(w.keywords) > 0:
                id_ += ', ' + ', '.join([k + '=' + str(v)
                                         for k, v in w.keywords.items()])
            yield id_

    # ensure args, keywords and ids are iterators
    if args is None:
        args = default_args()
    elif isinstance(args, (list, tuple)):
        args = iter(args)
    if keywords is None:
        keywords = default_keywords()
    elif isinstance(keywords, (list, tuple)):
        keywords = iter(keywords)
    if ids is None:
        ids = default_ids()
    elif isinstance(ids, (list, tuple)):
        ids = iter(ids)

    timings = collections.OrderedDict()
    memorys = collections.OrderedDict()

    while True:

        try:
            arg = next(args)
            keyword = next(keywords)
            id_ = str(next(ids))
        except StopIteration:
            break

        if not isinstance(arg, (list, tuple, str)):
            raise TypeError('The function arguments must be supplied as a seque'
                            'nce.')
        if not isinstance(keyword, dict):
            raise TypeError('The function keywords must be supplied as a dict.')

            
        stmt_ = stmt
        if isinstance(stmt, str) and isinstance(arg, str):
            while '*args' in stmt_:
                stmt_ = stmt_.replace('*args', arg)
        setup_ = setup
        if isinstance(setup, str) and isinstance(arg, str):
            while '*args' in setup_:
                setup_ = setup_.replace('*args', arg)

        if callable(stmt):
            w.args = arg
            w.keywords = keyword
            t = timeit.Timer(w, setup=setup_)
        else:
            t = timeit.Timer(stmt_, setup=setup_)

        # determine number so that 0.2 <= total time < 2.0
        for i in range(10):
            number = 10**i
            x = t.timeit(number)
            if x >= 0.2:
                break

        # actual runs
        gc.collect()
        memory = memory_usage()
        if number > 1 or x <= 2:
            r = t.repeat(repeat, number)
        else:
            r = t.repeat(repeat-1, number)
            r = [x] + r
        memory = memory_usage(since=memory)
        memorys[id_] = memory
        best = min(r)
        timings[id_] = best / number

        if id_ != '':
            id_ += ': '
        print id_ + "%d loops," % number,
        usec = best * 1e6 / number
        if usec < 1000:
            print "best of %d: %.*g us per loop. " % (repeat,precision,usec),
        else:
            msec = usec / 1000
            if msec < 1000:
                print "best of %d: %.*g ms per loop. " %(repeat,precision,msec),
            else:
                sec = msec / 1000
                print "best of %d: %.*g s per loop. " % (repeat,precision,sec),

        print ', '.join([k + ':' + str(v) + 'MiB' for k,v in memory.items()])
        if single_test:
            break

    results = timings, memorys
    if filename is not None:
        with open(filename, 'w') as f:
            pickle.dump(results, f)

    return results

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
    >>> cast([[1.,2.], None, array(2j)])
    (array([ 1.+0.j,  2.+0.j]), None, array(2j))

    """
    arrays = tuple(arrays)
    if dtype is None:
        arrays_ = [np.array(a, copy=False) for a in arrays if a is not None]
        dtype = np.result_type(*arrays_)
    result = (np.array(a, dtype=dtype, order=order, copy=False)
              if a is not None else None for a in arrays)
    return tuple(result)

def find(l, f):
    """
    Return first item in list that verifies a certain condition, or raise
    a ValueError exception otherwise.

    Parameters
    ----------
    l : list
        List of elements to be searched for.
    f : function
        Function that evaluates to True to match an element.

    Example:
    --------
    >>> find([1.,2.,3.], lambda x: x > 1.5)
    2.

    """
    try:
        return next((_ for _ in l if f(_)))
    except StopIteration:
        raise ValueError('There is no matching item in the list.')

def first_is_not(l, v):
    """
    Return first item in list which is not the specified value.
    If all items are the specified value, return it.
    """
    return next((_ for _ in l if _ is not v), v)

def ifind(l, f):
    """
    Return the number of the first item in a list that verifies a certain
    condition or raise a ValueError exception otherwise.

    Parameters
    ----------
    l : list
        List of elements to be searched for.
    f : function
        Function that evaluates to True to match an element.

    Example:
    --------
    >>> ifind([1.,2.,3.], lambda x: x > 1.5)
    1

    """
    try:
        return next((i for i, _ in enumerate(l) if f(_)))
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
    """ Make a block of code interruptible with CTRL-C. """
    signal_old = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, signal_old)

@contextmanager
def interruptible_if(condition):
    """ Conditionally make a block of code interruptible with CTRL-C. """
    if not condition:
        yield
    else:
        with interruptible():
            yield

def isalias(array1, array2):
    """
    Return True if the two input arrays point to the same memory location.

    """
    return array1.__array_interface__['data'][0] == \
           array2.__array_interface__['data'][0]

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
    it = np.nditer([a, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    max_power = [int(np.ceil(np.log(np.max(a))/np.log(v))) for v in l]
    slices = [slice(0, m+1) for m in max_power]
    powers = np.ogrid[slices]
    values = 1
    for v, p in izip(l, powers):
        values = values * v**p
    for v, o in it:
        if np.__version__ >= '1.8':
            o[...] = np.amin(values, where=values>=v)
        else:
            values_ = np.ma.MaskedArray(values, mask=values<v, copy=False)
            o[...] = np.min(values_)
    out = it.operands[1]
    if out.ndim == 0:
        return out.flat[0]
    return out

def memory_usage(keys=('VmRSS', 'VmData', 'VmSize'), since=None):
    """
    Return a dict containing information about the process' memory usage.

    Parameters
    ----------
    keys : sequence of strings
        Process status identifiers (see /proc/###/status). Default are
        the resident, data and virtual memory sizes.
    since : dict
        Dictionary as returned by a previous call to memory_usage function and
        used to compute the difference of memory usage since then.
        
    """
    proc_status = '/proc/%d/status' % os.getpid()
    scale = {'kB': 1024, 'mB': 1024*1024,
             'KB': 1024, 'MB': 1024*1024}

    # get pseudo file  /proc/<pid>/status
    with open(proc_status) as f:
        status = f.read()

    result = {}
    for k in keys:
        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = status.index(k)
        v = status[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            raise ValueError('Invalid format.')

        # convert Vm value to Mbytes
        result[k] = float(v[1]) * scale[v[2]] / 2**20

    if since is not None:
        if not isinstance(since, dict):
            raise TypeError('The input is not a dict.')
        common_keys = set(result.keys())
        common_keys.intersection_update(since.keys())
        result = dict((k, result[k] - since[k]) for k in common_keys)

    return result

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
    if any(p != q for p,q in izip(a,b) if None not in (p,q)):
        raise ValueError('The input sequences have incompatible values.')
    return tuple(p if p is not None else q for p,q in izip(a,b))
    
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

def renumerate(l):
    """ Reversed enumerate. """
    return izip(xrange(len(l)-1, -1, -1), reversed(l))

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

@contextmanager
def uninterruptible():
    """
    Make a block of code uninterruptible with CTRL-C.
    The KeyboardInterrupt is re-raised after the block is executed.

    """
    signal_old = signal.getsignal(signal.SIGINT)
    #XXX the nonlocal Python3 would be handy here
    ctrlc_is_pressed = []
    def signal_handler(signal, frame):
        ctrlc_is_pressed.append(True)
    signal.signal(signal.SIGINT, signal_handler)
    yield
    signal.signal(signal.SIGINT, signal_old)
    if len(ctrlc_is_pressed) > 0:
        raise KeyboardInterrupt()

@contextmanager
def uninterruptible_if(condition):
    """ Conditionally make a block of code uninterruptible with CTRL-C. """
    if not condition:
        yield
    else:
        with uninterruptible():
            yield
