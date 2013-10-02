from pyoperators import Operator, OperatorFlags, decorators

flags = OperatorFlags._fields


def test_decorators():
    def func(flag):
        if flag in ('shape_input', 'shape_output', 'inplace_reduction'):
            return
        o1 = Operator(flags={flag: True})

        class O2(Operator):
            pass
        O2 = eval('decorators.' + flag + '(O2)')
        o2 = O2()
        assert o1.flags == o2.flags
    for flag in flags:
        yield func, flag
