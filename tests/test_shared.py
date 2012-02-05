import numpy as np
from pyoperators import Operator
from numpy.testing import assert_equal

class ndarray1(np.ndarray):
    pass
class ndarray2(np.ndarray):
    pass
attr1 = { 'attr1': True, 'attr2': True}
attr2 = { 'attr1': False, 'attr3': False}

class ExplExpl(Operator):
    def __init__(self, shapein=3, shapeout=4, **keywords):
        Operator.__init__(self, shapein=shapein, shapeout=shapeout,
                          classout=ndarray1, attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:3] = input
        output[3] = 10.
class UncoExpl(Operator):
    def __init__(self, shapein=3, **keywords):
        Operator.__init__(self, shapein=shapein, classout=ndarray1,
                          attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:3] = 2*input
        output[3:] = 20
class ImplImpl(Operator):
    def __init__(self, **keywords):
        Operator.__init__(self, classout=ndarray1, attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:input.size] = 3*input
        output[-1] = 30
    def reshapein(self, shapein):
        return (shapein[0] + 1,)
    def reshapeout(self, shapeout):
        return (shapeout[0] - 1,)
class UncoImpl(Operator):
    def __init__(self, **keywords):
        Operator.__init__(self, classout=ndarray1, attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:output.size-1] = 4*input
        output[-1] = 40
    def reshapeout(self, shapeout):
        return (shapeout[0] - 1,)
class ExplUnco(Operator):
    def __init__(self, shapeout=4, **keywords):
        Operator.__init__(self, shapeout=shapeout, classout=ndarray1,
                          attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:3] = 5*input[0:3]
        output[3] = 50
class ImplUnco(Operator):
    def __init__(self, **keywords):
        Operator.__init__(self, classout=ndarray1, attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:input.size] = 6*input
        output[-1] = 60
    def reshapein(self, shapein):
        return (shapein[0] + 1,)
class UncoUnco(Operator):
    def __init__(self, **keywords):
        Operator.__init__(self, classout=ndarray1, attrout=attr1, **keywords)
    def direct(self, input, output):
        output[0:3] = 7*input[0:3]
        output[3:] = 70

Ops = ExplExpl, UncoExpl, ImplImpl, UncoImpl, ExplUnco, ImplUnco, UncoUnco
ops = tuple(cls() for cls in Ops)

def test():
    kind = {'Expl':'explicit', 'Impl': 'implicit', 'Unco':'unconstrained'}
    def func(flags, name):
        assert_equal(flags.shape_output, kind[name[:4]])
        assert_equal(flags.shape_input, kind[name[4:]])

    for op in ops:
        yield func, op.flags, type(op).__name__
