import nose
from nose.tools import (eq_, ok_, assert_is_none, assert_is_not_none, assert_is,
                        assert_is_not, assert_not_in, assert_is_instance, nottest)
import numpy as np
from numpy.testing import assert_array_equal

from operators.core import Operator, AdditionOperator, CompositionOperator, ScalarOperator, ndarraywrap
from operators.linear import I, O
from operators.decorators import symmetric, square

def assert_flags(operator, flags):
    if isinstance(flags, str):
        flags = flags.split(',')
    for f in flags:
        assert getattr(operator.flags, f.replace(' ', ''))


dtypes = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64,
          np.float128, np.complex64, np.complex128, np.complex256)]

class ndarray2(np.ndarray):
    pass

class ndarray3(np.ndarray):
    pass

class ndarray4(np.ndarray):
    pass

@square
class I2(Operator):
    def direct(self, input, output):
        output.newattr = True
    def transpose(self, input, output):
        pass            

@square
class I3(Operator):
    def direct(self, input, output):
        output.__class__ = ndarray3
    def transpose(self, input, output):
        output.__class__ = ndarray4

def test_shapes():
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3,2], np.array([3,2]), (3,2),
                    [3.,2], np.array([3.,2]), (3.,2),
                   ):
        o = Operator(shapein=shapein)
        yield assert_is, type(o.shapein), tuple
        yield ok_, all([isinstance(s, int) for s in o.shapein])


def test_dtype1():
    value = 2.5
    class Op(Operator):
        def __init__(self, dtype):
            Operator.__init__(self, dtype=dtype)
        def direct(self, input, output):
            np.multiply(input, np.array(value, self.dtype), output)
    input = complex(1,1)
    for dop in dtypes:
        for di in dtypes:
            try:
                i = np.array(input, di)
            except TypeError:
                i = np.array(input.real, di)
            o = Op(dop)(i)
            yield eq_, o.dtype, (i * np.array(value, dop)).dtype
            yield assert_array_equal, o, i * np.array(value, dop)


def test_dtype2():
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)
    op = Op()
    input = complex(1,1)
    for di in dtypes:
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = op(i)
        yield eq_, o.dtype, (i * i).dtype
        yield assert_array_equal, o, i * i


def test_symmetric():    
    mat = np.matrix([[2,1],[1,2]])
    @symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=(2,), dtype=mat.dtype)
        def direct(self, input, output):
            output[:] = np.dot(mat, input)

    op = Op()
    yield assert_flags, op, 'LINEAR,SQUARE,REAL,SYMMETRIC'
    yield eq_, op.shape, (2,2)
    yield eq_, op.shapeout, (2,)
    yield assert_is, op, op.C
    yield assert_is, op, op.T
    yield assert_is, op, op.H
    yield assert_array_equal, op([1,1]), np.array(mat * [[1],[1]]).ravel()


def test_scalar_reduction1():
    models = 1.+I, -I, (-2) * I, -(2*I), 1.-I, 1.-2*I
    results = [6, -3, -6, -6, 0, -3]
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            o = model(i)
            yield eq_, o, result
            yield eq_, o.dtype, np.float64


def test_scalar_reduction2():
    model = -I
    iops = '__iadd__', '__isub__', '__imul__', '__iadd__', '__imul__'
    imodels = 2*I, 2*I, 2, O, O
    results = [3, -3, -6, -6, 0]
    for iop, imodel, result in zip(iops, imodels, results):
        model = getattr(model, iop)(imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield eq_, model(i), result


def test_propagation_attribute():
    @square
    class AddAttribute(Operator):
        def direct(self, input, output):
            output.newattr_direct = True
        def transpose(self, input, output):
            output.newattr_transpose = True

    @square
    class AddAttribute2(Operator):
        def direct(self, input, output):
            output.newattr_direct = False
        def transpose(self, input, output):
            output.newattr_transpose = False

    @square
    class AddAttribute3(Operator):
        def direct(self, input, output):
            output.newattr3_direct = True
        def transpose(self, input, output):
            output.newattr3_transpose = True

    inputs = [np.ones(5), np.ones(5).view(ndarray2)]
    for i in inputs:
        op = AddAttribute()
        yield ok_, op(i).newattr_direct
        yield ok_, op.T(i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        yield ok_, not op(i).newattr_direct
        yield ok_, op.T(i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        yield ok_, op(i).newattr_direct
        yield ok_, op(i).newattr3_direct
        yield ok_, op.T(i).newattr_transpose
        yield ok_, op.T(i).newattr3_transpose

    for i_ in inputs:
        op = AddAttribute()
        i = i_.copy()
        yield ok_, op(i,i).newattr_direct
        if type(i) is not np.ndarray:
            yield ok_, i.newattr_direct
        i = i_.copy()
        yield ok_, op.T(i,i).newattr_transpose
        if type(i) is not np.ndarray:
            yield ok_, i.newattr_transpose

        op = AddAttribute2() * AddAttribute()
        i = i_.copy()
        yield ok_, not op(i,i).newattr_direct
        if type(i) is not np.ndarray:
            yield ok_, not i.newattr_direct
        i = i_.copy()
        yield ok_, op.T(i,i).newattr_transpose
        if type(i) is not np.ndarray:
            yield ok_, i.newattr_transpose

        op = AddAttribute3() * AddAttribute()
        i = i_.copy()
        o = op(i,i)
        yield ok_, o.newattr_direct
        yield ok_, o.newattr3_direct
        if type(i) is not np.ndarray:
            yield ok_, i.newattr_direct
            yield ok_, i.newattr3_direct
        i = i_.copy()
        o = op.T(i,i)
        yield ok_, o.newattr_transpose
        yield ok_, o.newattr3_transpose
        if type(i) is not np.ndarray:
            ok_, i.newattr_transpose
            ok_, i.newattr3_transpose


def check_propagation_class(op, i, c):
    o = op(i)
    assert_is(o.__class__, c)

def check_propagation_class_inplace(op, i, c):
    i = i.copy()
    if type(i) is np.ndarray:
        op(i,i)
        assert_is(i.__class__, np.ndarray)
    else:
        op(i,i)
        assert_is(i.__class__, c)

def test_propagation_class():

    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, I2(), I2()*I3(), I3()*I2()]
    results = [[np.ndarray, ndarray2],
               [ndarraywrap, ndarray2],
               [ndarray3, ndarray3],
               [ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op, i, c

def test_propagation_class_inplace():

    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, I2(), I2()*I3(), I3()*I2()]
    results = [[np.ndarray, ndarray2],
               [ndarraywrap, ndarray2],
               [ndarray3, ndarray3],
               [ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op, i, c

def test_propagation_classT():

    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, I2(), I2()*I3(), I3()*I2()]
    resultsT = [[np.ndarray, ndarray2],
                [np.ndarray, ndarray2],
                [ndarray4, ndarray4],
                [ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op.T, i, c

def test_propagation_classT_inplace():

    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, I2(), I2()*I3(), I3()*I2()]
    resultsT = [[np.ndarray, ndarray2],
                [np.ndarray, ndarray2],
                [ndarray4, ndarray4],
                [ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op.T, i, c

def test_propagation_class_nested():
    class O1(Operator):
        def direct(self, input, output):
            output[:] = input
            output.__class__ = ndarray2            
    class O2(Operator):
        def direct(self, input, output):
            output[:] = input
    o1 = O1()
    o2 = O2()
    ops1 = [I, 2, o2, 2 + o2]
    ops2 = [1+o1, 2*o1, o1+o2, o2+o1, 1+2*o1, 1+o1+o2, 1+o2+o1, o1+1+o2,
            o1+o2+1, o2+o1+1, o2+1+o1]
    for op1 in ops1:
        for op2 in ops2:
            yield assert_is, (op1*op2)(1).__class__, ndarray2
            yield assert_is, (op2*op1)(1).__class__, ndarray2
    for op1 in ops1:
        for op2 in ops2:
            for op3 in ops1:
                yield assert_is, (op1*op2*op3)(1).__class__, ndarray2
        
def test_addition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.sum([Op(v) for v in [1]])
    yield assert_is, op.__class__, Op

    op = np.sum([Op(v) for v in [1,2]])
    yield eq_, op.__class__, AdditionOperator
    yield assert_array_equal, op(1), 3
    yield assert_is_not_none, op.work[0]
    yield assert_is_none, op.work[1]

    op = np.sum([Op(v) for v in [1,2,4]])
    yield assert_is, op.__class__, AdditionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    yield assert_array_equal, op(input, output), 7
    yield assert_array_equal, input, 1
    yield assert_array_equal, output, 7
    yield assert_is_not_none, op.work[0]
    yield assert_is_none, op.work[1]

    output = input
    yield assert_array_equal, op(input, output), 7
    yield assert_array_equal, input, 7
    yield assert_array_equal, output, 7
    yield assert_is_not_none, op.work[0]
    yield assert_is_not_none, op.work[1]


def test_composition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.product([Op(v) for v in [1]])
    yield assert_is, op.__class__, Op

    op = np.product([Op(v) for v in [1,2]])
    yield assert_is, op.__class__, CompositionOperator
    yield assert_array_equal, op(1), 2
    yield assert_is_none, op.work[0]
    yield assert_is_none, op.work[1]

    op = np.product([Op(v) for v in [1,2,4]])
    yield assert_is, op.__class__, CompositionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    yield assert_array_equal, op(input, output), 8
    yield assert_array_equal, input, 1
    yield assert_array_equal, output, 8
    yield assert_is_none, op.work[0]
    yield assert_is_none, op.work[1]

    output = input
    yield assert_array_equal, op(input, output), 8
    yield assert_array_equal, input, 8
    yield assert_array_equal, output, 8
    yield assert_is_none, op.work[0]
    yield assert_is_none, op.work[1]


def test_scalar_operator():
    s = ScalarOperator(1)
    yield ok_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(-1)
    yield ok_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(2.)
    yield ok_, s.C is s.T is s.H is s
    yield assert_is_not, s.I, s
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_is_instance, o, ScalarOperator

    s = ScalarOperator(complex(0,1))
    yield assert_is, s.T, s
    yield assert_is, s.H, s.C
    yield assert_not_in, s.I, (s, s.C)
    yield assert_not_in, s.I.C, (s, s.C)
    yield assert_is_instance, s.C, ScalarOperator
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_is_instance, o, ScalarOperator


if __name__ == "__main__":
    nose.run(argv=['', __file__])
