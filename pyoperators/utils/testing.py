import functools
import numpy as np
from nose.plugins.skip import SkipTest
from numpy.testing import assert_equal

from .misc import all_eq, strenum

__all__ = ['assert_eq',
           'assert_in',
           'assert_not_in',
           'assert_is',
           'assert_is_not',
           'assert_is_instance',
           'assert_is_not_instance',
           'assert_is_none',
           'assert_is_not_none',
           'assert_is_type',
           'assert_raises',
           'skiptest',
           'skiptest_if',
           'skiptest_unless_module']


def assert_same(actual, desired, rtol=2, broadcasting=False):
    """
    Compare arrays of floats. The relative error depends on the data type.

    Parameters
    ----------
    rtol : float
        Relative tolerance to account for numerical error propagation.
    broadcasting : bool, optional
        If true, allow broadcasting betwee, actual and desired array.

    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if not broadcasting and actual.shape != desired.shape:
        raise AssertionError(
            "The actual array shape '{0}' is different from the desired one '{"
            "1}'.".format(actual.shape, desired.shape))
    dtype = sorted(arg.dtype for arg in [actual, desired])[0]
    if dtype.kind in ('b', 'i'):
        if not broadcasting:
            assert_equal(actual, desired)
        else:
            assert np.all(actual == desired)
        return
    eps = np.finfo(dtype).eps * rtol
    same = abs(actual - desired) <= eps * np.maximum(abs(actual), abs(desired))
    same |= np.isnan(actual) & np.isnan(desired)
    if not np.all(same):
        def trepr(x):
            r = repr(x).split('\n')
            if len(r) > 3:
                r = [r[0], r[1], r[2] + ' ...']
            return '\n'.join(r)
        raise AssertionError(
            "Arrays are not equal.\n\n(mismatch {0:.1%})\n x: {1}\n y: {2}"
            .format(1 - np.mean(same), trepr(actual), trepr(desired)))


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
    """
    Assert that the first argument is not an instance of the second one.

    """
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


def assert_is_type(a, cls, msg=None):
    """ Assert argument is of a specified type. """
    if type(cls) is type:
        cls = (cls,)
    else:
        cls = tuple(cls)
    if any(type(a) is t for t in cls):
        return
    raise AssertionError(
        "{0} is of type '{1}' instead of {2}{3}".format(
        a, type(a).__name__, strenum(c.__name__ for c in cls), _get_msg(msg)))


def assert_raises(*args, **kwargs):
    np.testing.assert_raises(*args, **kwargs)
assert_raises.__doc__ = np.testing.assert_raises.__doc__


def skiptest(func):
    @functools.wraps(func)
    def _():
        raise SkipTest()
    return _


def skiptest_if(condition):
    def decorator(func):
        @functools.wraps(func)
        def _():
            if condition:
                raise SkipTest()
            func()
        return _
    return decorator


def skiptest_unless_module(module):
    def decorator(func):
        @functools.wraps(func)
        def _():
            try:
                __import__(module)
            except ImportError:
                raise SkipTest()
            func()
        return _
    return decorator


def _get_msg(msg):
    if not msg:
        return '.'
    return ': ' + str(msg) + '.'
