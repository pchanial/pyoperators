import pytest

import pyoperators.flags
from pyoperators import Operator


@pytest.mark.parametrize('flag', dir(pyoperators.flags))
def test_flags(flag):
    if flag not in pyoperators.flags.Flags._fields:
        return
    if flag in ('shape_input', 'shape_output'):
        return
    if flag == 'outplace':
        flags = {'outplace': False}
    else:
        flags = {flag: True}
    o1 = Operator(flags=flags)

    class O2(Operator):
        pass

    O2 = eval('pyoperators.flags.' + flag + '(O2)')
    o2 = O2()
    assert o1.flags == o2.flags
