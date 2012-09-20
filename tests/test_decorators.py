from pyoperators import Operator, OperatorFlags, decorators

flags = OperatorFlags._fields
    
def test_decorators():
    def func(flag):
        if flag in ('shape_input', 'shape_output', 'inplace_reduction'):
            return
        value = True if flag not in ('alignment_input', 'alignment_output') \
                else 32
        o1 = Operator(flags={flag:value})
        class O2(Operator):
            pass
        if flag == 'alignment_input':
            flag = 'aligned_input'
        elif flag == 'alignment_output':
            flag = 'aligned_output'
        O2 = eval('decorators.' + flag + '(O2)')
        o2 = O2()
        assert o1.flags == o2.flags
    for flag in flags:
        yield func, flag
