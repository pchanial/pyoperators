import nose
from nose.tools import (eq_, ok_, assert_is_none, assert_is_not_none, assert_is,
                        assert_is_not, assert_not_in, assert_is_instance)
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_raises

from operators.core import Operator, AdditionOperator, CompositionOperator, PartitionOperator, ScalarOperator, ndarraywrap
from operators.linear import I, O, DiagonalOperator, IdentityOperator
from operators.decorators import symmetric, square

np.seterr(all='raise')

def assert_flags(operator, flags, msg=''):
    if isinstance(flags, str):
        flags = [f.replace(' ', '') for f in flags.split(',')]
    for f in flags:
        assert getattr(operator.flags, f), 'Operator {0} is not {1}.'.format(
            operator, f) + (' ' + msg if msg else '')

def assert_flags_false(operator, flags, msg=''):
    if isinstance(flags, str):
        flags = [f.replace(' ', '') for f in flags.split(',')]
    for f in flags:
        assert not getattr(operator.flags, f),'Operator {0} is not {1}.'.format(
            operator, f) + (' ' + msg if msg else '')

def assert_is_inttuple(shape, msg=''):
    msg = '{0} is not an int tuple.'.format(shape) + (' ' + msg if msg else '')
    assert type(shape) is tuple, msg
    assert all([isinstance(s, int) for s in shape]), msg

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

def test_shape_is_inttuple():
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3,2], np.array([3,2]), (3,2),
                    [3.,2], np.array([3.,2]), (3.,2)):
        o = Operator(shapein=shapein, shapeout=shapein)
        yield assert_is_inttuple, o.shapein, 'shapein: '+str(shapein)
        yield assert_is_inttuple, o.shapeout, 'shapeout: '+str(shapein)

def test_shape_explicit():
    o1, o2, o3 = (
        Operator(shapeout=(13,2), shapein=(2,2)),
        Operator(shapeout=(2,2), shapein=(1,3)),
        Operator(shapeout=(1,3), shapein=4))
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((13,2),(2,2),(13,2)),
                            ((1,3),(4,),(4,))):
        yield assert_equal, o.shapeout, eout, '*shapeout:'+str(o)
        yield assert_equal, o.shapein, ein, '*shapein:'+str(o)
    yield assert_raises, ValueError, CompositionOperator, [o2, o1]
    yield assert_raises, ValueError, CompositionOperator, [o3, o2]
    yield assert_raises, ValueError, CompositionOperator, [o3, I, o1]

    o4 = Operator(shapeout=o1.shapeout)
    o5 = Operator(flags={'SQUARE':True})

    o1 = Operator(shapein=(13,2))
    for o in [o1+I, I+o1, o1+o4, o1+I+o5+o4, I+o5+o1]:
        yield assert_equal, o.shapein, o1.shapein, '+shapein:'+str(o)
        yield assert_equal, o.shapeout, o1.shapeout, '+shapeout:'+str(o)
    yield assert_raises, ValueError, AdditionOperator, [o2, o1]
    yield assert_raises, ValueError, AdditionOperator, [o3, o2]
    yield assert_raises, ValueError, AdditionOperator, [I, o3, o1]
    yield assert_raises, ValueError, AdditionOperator, [o3, I, o1]

def test_shape_implicit():
    class Op(Operator):
        def __init__(self, factor):
            self.factor = factor
            Operator.__init__(self)
        def reshapein(self, shape):
            return shape[0]*self.factor
        def reshapeout(self, shape):
            return shape[0]/self.factor
        def __str__(self):
            return super(Op, self).__str__() + ' {0}'.format(self.factor)
    o1, o2, o3 = (Op(2), Op(3), Op(4))
    assert o1.shapein is o2.shapein is o3.shapein is None
    shapein = (1,)
    shapeout = (24,)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((6,),(12,),(24,)),
                            ((4,),(2,),(1,))):
        yield assert_equal, o._reshapein(shapein), eout, 'reshapein:'+str(o)
        yield assert_equal, o._reshapeout(shapeout), ein, 'reshapeout:'+str(o)

def test_shape_cornercases():
    op = Operator()
    assert_flags(op, 'SQUARE')
    op = Operator(shapein=2)
    assert_flags(op, 'SQUARE')
    assert op.shapeout == op.shapein

    class Op(Operator):
        def reshapein(self, shape):
            return shape[0]*2
    op = Op()
    assert_flags_false(op, 'SQUARE')
    op = Op(shapein=2)
    assert_flags_false(op, 'SQUARE')
    assert op.shapeout == (4,)
    
    assert_raises(ValueError, Op, shapein=3, shapeout=11)

    op = Op(shapein=2) * Operator(shapein=4, shapeout=2)
    assert_flags(op, 'SQUARE')

def test_dtype1():
    value = 2.5
    @square
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
            yield assert_equal, o.dtype, (i*np.array(value,dop)).dtype, str((dop,di))
            yield assert_array_equal, o, i*np.array(value,dop), str((dop,di))


def test_dtype2():
    @square
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
        yield assert_equal, o.dtype, (i * i).dtype, str(di)
        yield assert_array_equal, o, i * i, str(di)


def test_symmetric():    
    mat = np.matrix([[2,1],[1,2]])
    @symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=(2,), dtype=mat.dtype)
        def direct(self, input, output):
            output[:] = np.dot(mat, input)

    op = Op()
    assert_flags(op, 'LINEAR,SQUARE,REAL,SYMMETRIC')
    assert_equal(op.shape, (2,2))
    assert_equal(op.shapeout, (2,))
    assert op is op.C
    assert op is op.T
    assert op is op.H
    assert_array_equal(op([1,1]), np.array(mat * [[1],[1]]).ravel())


def test_scalar_reduction1():
    models = 1.+I, -I, (-2) * I, -(2*I), 1.-I, 1.-2*I
    results = [6, -3, -6, -6, 0, -3]
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            o = model(i)
            yield assert_equal, o, result, str((model,i))
            yield assert_equal, o.dtype, np.float64, str((model,i))


def test_scalar_reduction2():
    model = -I
    iops = '__iadd__', '__isub__', '__imul__', '__iadd__', '__imul__'
    imodels = 2*I, 2*I, 2, O, O
    results = [3, -3, -6, -6, 0]
    for iop, imodel, result in zip(iops, imodels, results):
        model = getattr(model, iop)(imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield eq_, model(i), result, str((iop,i))


def test_scalar_reduction3():
    for opout in (None, 100, 200):
        for opin in (None, 100, 200):
            for idin in (None, 100, 200):
                if opin is not None and idin is not None and opin != idin:
                    continue
                print opout, opin, idin
                p = Operator(shapeout=opout, shapein=opin,
                    flags={'LINEAR':True}) * IdentityOperator(shapein=idin)

                n = len(p.operands) if isinstance(p, CompositionOperator) else 1
                if idin is None:
                    yield assert_equal, n, 1, str((opout,opin,idin,p))
                    continue
                if opin is None:
                    yield assert_equal, n, 2, str((opout,opin,idin,p))
                    yield ok_, isinstance(p.operands[1], IdentityOperator), \
                        str((opout,opin,idin,p))
                    continue
                if opin == idin and opout == idin:
                    yield assert_equal, n, 1, str((opout,opin,idin,p))


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
    @square
    class O1(Operator):
        def direct(self, input, output):
            output[:] = input
            output.__class__ = ndarray2            
    @square
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
    @square
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
    @square
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

def test_partition1():
    o1 = ScalarOperator(1, shapein=1)
    o2 = ScalarOperator(2, shapein=2)
    o3 = ScalarOperator(3, shapein=3)
    
    r = DiagonalOperator([1,2,2,3,3,3]).todense()
    for ops, p in zip(((o1,o2,o3), (I,o2,o3), (o1,2*I,o3), (o1,o2,3*I)),
                      (None, (1,2,3), (1,2,3), (1,2,3))):
        op = PartitionOperator(ops, partitionin=p)
        yield assert_array_equal, op.todense(6), r, str(op)

def test_partition2():
    class Op(Operator):
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
    i = np.arange(3*4*5*6).reshape(3,4,5,6)
    for axisp,p in zip((0,1,2,3,-1,-2,-3), ((1,1,1),(1,2,1),(2,2,1),(2,3,1), (2,3,1),(2,2,1), (1,2,1), (1,1,1))):
        for axisr in (0,1,2,3):
            op = PartitionOperator(3*[Op(axisr)], partitionin=p, axisin=axisp)
            yield assert_array_equal, op(i), Op(axisr)(i), 'axis={},{}'.format(
                axisp,axisr)

def test_partition3():
    # test axisin != axisout...
    pass


def test_partition4():
    o1 = ScalarOperator(1, shapein=1)
    o2 = ScalarOperator(2, shapein=2)
    o3 = ScalarOperator(3, shapein=3)
    class Op(Operator):
        pass
    op=Op()
    p=PartitionOperator([o1,o2,o3])
    
    r = (op + p + op) * p
    assert isinstance(r, PartitionOperator)

if __name__ == "__main__":
    nose.run(argv=['', __file__])
