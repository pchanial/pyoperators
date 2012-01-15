from pyoperators import Operator, OperatorFlags, decorators

flags = OperatorFlags._fields


def test_decorators():
    for flag in flags:
        if flag in ('shape_input', 'shape_output'):
            continue
        o1 = Operator(flags={flag: True})

        class O2(Operator):
            pass

        O2 = eval('decorators.' + flag.lower() + '(O2)')
        o2 = O2()
        assert o1.flags == o2.flags
