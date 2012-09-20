import numpy as np
import pyoperators

from pyoperators import Operator, decorators
from pyoperators.utils.testing import assert_eq

DTYPES = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64,
          np.float16, np.float32, np.float64, np.float128,
          np.complex64, np.complex128, np.complex256)]

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

OPS = ExplExpl, UncoExpl, ImplImpl, UncoImpl, ExplUnco, ImplUnco, UncoUnco

ALL_OPS = [ eval('pyoperators.' + op) for op in dir(pyoperators) if op.endswith(
            'Operator') ]

@decorators.square
class IdentityOutplaceOperator(Operator):
    def direct(self, input, output):
        output[...] = input

@decorators.real
@decorators.symmetric
class HomothetyOutplaceOperator(Operator):
    def __init__(self, value, **keywords):
        Operator.__init__(self, **keywords)
        self.value = value
    def direct(self, input, output):
        output[...] = self.value * input


class Stretch(Operator):
    """ Stretch input array by replicating it by a factor of 2. """
    def __init__(self, axis, **keywords):
        self.axis = axis
        if self.axis < 0:
            self.slice = [Ellipsis] + (-self.axis) * [slice(None)]
        else:
            self.slice = (self.axis+1) * [slice(None)] + [Ellipsis]
        Operator.__init__(self, **keywords)
    def direct(self, input, output):
        self.slice[self.axis] = slice(0,None,2)
        output[self.slice] = input
        self.slice[self.axis] = slice(1,None,2)
        output[self.slice] = input
    def reshapein(self, shape):
        shape_ = list(shape)
        shape_[self.axis] *= 2
        return shape_
    def reshapeout(self, shape):
        shape_ = list(shape)
        shape_[self.axis] //= 2
        return shape_

def assert_inplace_outplace(op, v, expected):
    w = op(v)
    assert_eq(w, expected)
    op(v, out=w)
    assert_eq(w, expected)
