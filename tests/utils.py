def assert_in(a, b, msg=None):
    if a in b:
        return
    assert False, str(a) + ' is not in ' + str(b) + _get_msg(msg)

def assert_not_in(a, b, msg=None):
    if a not in b:
        return
    assert False, str(a) + ' is in ' + str(b) + _get_msg(msg)

def assert_is(a, b, msg=None):
    if a is b:
        return
    assert False, str(a) + ' is not ' + str(b) + _get_msg(msg)

def assert_is_not(a, b, msg=None):
    if a is not b:
        return
    assert False, str(a) + ' is ' + str(b) + _get_msg(msg)

def assert_is_instance(a, cls, msg=None):
    if isinstance(a, cls):
        return
    assert False, str(a) + " is not a '" + cls.__name__ + "' instance" + \
           _get_msg(msg)

def assert_is_not_instance(a, cls, msg=None):
    if not isinstance(a, cls):
        return
    assert False, str(a) + " is a '" + cls.__name__ + "' instance" + \
           _get_msg(msg)

def assert_is_none(a, msg=None):
    if a is None:
        return
    assert False, str(a) + ' is not None' + _get_msg(msg)

def assert_is_not_none(a, msg=None):
    if a is not None:
        return
    assert False, str(a) + ' is None' + _get_msg(msg)

def _get_msg(msg):
    if not msg:
        return '.'
    return ': ' + str(msg) + '.'
