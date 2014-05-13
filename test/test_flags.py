import pyoperators


def test_flags():
    def func(flag):
        if flag not in pyoperators.flags.Flags._fields:
            return
        if flag in ('shape_input', 'shape_output'):
            return
        if flag == 'outplace':
            flags = {'outplace': False}
        else:
            flags = {flag: True}
        o1 = pyoperators.Operator(flags=flags)

        class O2(pyoperators.Operator):
            pass
        O2 = eval('pyoperators.flags.' + flag + '(O2)')
        o2 = O2()
        assert o1.flags == o2.flags
    for flag in dir(pyoperators.flags):
        yield func, flag

