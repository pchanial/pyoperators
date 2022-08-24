import functools
from collections.abc import Container, Mapping

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from .misc import settingerr, strenum

__all__ = [
    'assert_eq',
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
    'skiptest_unless_module',
]


def assert_same(actual, desired, atol=0, rtol=5, broadcasting=False):
    """
    Compare arrays of floats. The relative error depends on the data type.

    Parameters
    ----------
    atol : float
        Absolute tolerance to account for numerical error propagation, in
        unit of eps.
    rtol : float
        Relative tolerance to account for numerical error propagation, in
        unit of eps.
    broadcasting : bool, optional
        If true, allow broadcasting betwee, actual and desired array.

    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if actual.dtype.kind not in ('b', 'i', 'u', 'f', 'c') or desired.dtype.kind not in (
        'b',
        'i',
        'u',
        'f',
        'c',
    ):
        raise TypeError('Non numeric type.')
    if not broadcasting and actual.shape != desired.shape:
        raise AssertionError(
            f"The actual array shape '{actual.shape}' is different from the desired one"
            f" '{desired.shape}'."
        )
    if actual.dtype.kind in ('b', 'i', 'u') and desired.dtype.kind in ('b', 'i', 'u'):
        if not broadcasting:
            assert_equal(actual, desired)
        else:
            assert np.all(actual == desired)
        return
    if actual.dtype.kind in ('b', 'i', 'u'):
        dtype = desired.dtype
    elif desired.dtype.kind in ('b', 'i', 'u'):
        dtype = actual.dtype
    else:
        dtype = sorted(_.dtype for _ in (actual, desired))[0]

    eps1 = np.finfo(dtype).eps * rtol
    eps2 = np.finfo(dtype).eps * atol

    with settingerr('ignore'):
        same_ = (
            abs(actual - desired) <= eps1 * np.minimum(abs(actual), abs(desired)) + eps2
        )
        same = same_ | np.isnan(actual) & np.isnan(desired) | (actual == desired)
        if np.all(same):
            return

        msg = f'Arrays are not equal (mismatch {1-np.mean(same):.1%}'
        if np.any(~same_ & np.isfinite(actual) & np.isfinite(desired)):
            rtolmin = np.nanmax(
                abs(actual - desired) / np.minimum(abs(actual), abs(desired))
            )
            atolmin = np.nanmax(abs(actual - desired))
            min_rtol = rtolmin / np.finfo(dtype).eps
            min_atol = atolmin / np.finfo(dtype).eps
            msg += f', min rtol: {min_rtol}, min atol: {min_atol}'
        check_nan = np.isnan(actual) & ~np.isnan(desired) | np.isnan(
            desired
        ) & ~np.isnan(actual)
        if np.any(check_nan):
            msg += ', check nan'
        if np.any(
            ~check_nan & (np.isinf(actual) | np.isinf(desired)) & (actual != desired)
        ):
            msg += ', check infinite'

        def trepr(x):
            r = repr(x).split('\n')
            if len(r) > 3:
                r = [r[0], r[1], r[2] + ' ...']
            return '\n'.join(r)

        raise AssertionError(f'{msg})\n x: {trepr(actual)}\n y: {trepr(desired)}')


def assert_eq(a, b, msg=''):
    """Assert that the two arguments are equal."""
    if a is b:
        return

    if not msg:
        msg = f'Items are not equal:\n ACTUAL: {a}\n DESIRED: {b}'

    # a or b is an ndarray sub-class
    if (
        isinstance(a, np.ndarray)
        and type(a) not in (np.matrix, np.ndarray)
        or isinstance(b, np.ndarray)
        and type(b) not in (np.matrix, np.ndarray)
    ):
        assert_is(type(a), type(b))
        assert_allclose(a.view(np.ndarray), b.view(np.ndarray), err_msg=msg)
        assert_eq(a.__dict__, b.__dict__, msg)
        return

    # a and b are ndarray or one of them is an ndarray and the other is a seq.
    num_types = (bool, int, float, complex, np.ndarray, np.number)
    if (
        isinstance(a, num_types)
        and isinstance(b, num_types)
        or isinstance(a, np.ndarray)
        and isinstance(b, (list, tuple))
        or isinstance(b, np.ndarray)
        and isinstance(a, (list, tuple))
    ):
        assert_allclose(a, b, err_msg=msg)
        return

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        raise AssertionError(msg)

    if isinstance(a, Mapping) and isinstance(b, Mapping):
        assert_equal(set(a.keys()), set(b.keys()), err_msg=msg)
        for k in a:
            assert_eq(a[k], b[k], msg)
        return

    if (
        isinstance(a, Container)
        and not isinstance(a, (set, str))
        and isinstance(b, Container)
        and not isinstance(b, (set, str))
    ):
        assert_equal(len(a), len(b), msg)
        for a_, b_ in zip(a, b):
            assert_eq(a_, b_, msg)
        return

    try:
        equal = a == b
    except Exception:
        equal = False

    assert equal, msg


def assert_in(a, b, msg=None):
    """Assert that the first argument is in the second one."""
    if a in b:
        return
    assert False, f'{a} is not in {b}{_get_msg(msg)}'


def assert_not_in(a, b, msg=None):
    """Assert that the first argument is not in second one."""
    if a not in b:
        return
    assert False, f'{a} is in {b}{_get_msg(msg)}'


def assert_is(a, b, msg=None):
    """Assert arguments are equal as determined by the 'is' operator."""
    if a is b:
        return
    assert False, f'{a} is not {b}{_get_msg(msg)}'


def assert_is_not(a, b, msg=None):
    """Assert arguments are not equal as determined by the 'is' operator."""
    if a is not b:
        return
    assert False, f'{a} is {b}{_get_msg(msg)}'


def assert_is_instance(a, cls, msg=None):
    """Assert that the first argument is an instance of the second one."""
    if isinstance(a, cls):
        return
    assert False, f'{a} is not a {cls.__name__!r} instance{_get_msg(msg)}'


def assert_is_not_instance(a, cls, msg=None):
    """
    Assert that the first argument is not an instance of the second one.

    """
    if not isinstance(a, cls):
        return
    assert False, f'{a} is a {cls.__name__!r} instance{_get_msg(msg)}'


def assert_is_none(a, msg=None):
    """Assert argument is None."""
    if a is None:
        return
    assert False, f'{a} is not None{_get_msg(msg)}'


def assert_is_not_none(a, msg=None):
    """Assert argument is not None."""
    if a is not None:
        return
    assert False, f'{a} is None{_get_msg(msg)}'


def assert_is_type(a, cls, msg=None):
    """Assert argument is of a specified type."""
    if type(cls) is type:
        cls = (cls,)
    else:
        cls = tuple(cls)
    if any(type(a) is t for t in cls):
        return
    raise AssertionError(
        f'{a} is of type {type(a).__name__!r} instead of '
        f'{strenum(c.__name__ for c in cls)}{_get_msg(msg)}'
    )


def assert_raises(*args, **kwargs):
    np.testing.assert_raises(*args, **kwargs)


assert_raises.__doc__ = np.testing.assert_raises.__doc__


def skiptest(func):
    from nose.plugins.skip import SkipTest

    @functools.wraps(func)
    def _():
        raise SkipTest()

    return _


def skiptest_if(condition):
    from nose.plugins.skip import SkipTest

    def decorator(func):
        @functools.wraps(func)
        def _():
            if condition:
                raise SkipTest()
            func()

        return _

    return decorator


def skiptest_unless_module(module):
    from nose.plugins.skip import SkipTest

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
