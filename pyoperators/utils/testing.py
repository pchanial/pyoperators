import numpy as np
from .misc import all_eq

__all__ = ['assert_eq',
           'assert_in',
           'assert_not_in',
           'assert_is',
           'assert_is_not',
           'assert_is_instance',
           'assert_is_not_instance',
           'assert_is_none',
           'assert_is_not_none',
           'assert_raises',
           'skiptest']

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

def assert_raises(*args, **kwargs):
    np.testing.assert_raises(*args, **kwargs)
assert_raises.__doc__ = np.testing.assert_raises.__doc__

def skiptest(func):
    from nose.plugins.skip import SkipTest
    def _():
        raise SkipTest()
    _.__name__ = func.__name__
    return _

def _get_msg(msg):
    if not msg:
        return '.'
    return ': ' + str(msg) + '.'
