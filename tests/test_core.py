import itertools
import numpy as np
import operator
import pyoperators

from pyoperators import memory, decorators
from pyoperators.core import (Operator, AdditionOperator, BroadcastingOperator,
         BlockOperator, BlockColumnOperator, BlockDiagonalOperator,
         BlockRowOperator, BlockSliceOperator, CompositionOperator,
         ConstantOperator, DiagonalOperator, IdentityOperator,
         MultiplicationOperator, HomothetyOperator, ZeroOperator, asoperator, I,
         O)
from pyoperators.utils import ndarraywrap, merge_none
from pyoperators.utils.mpi import MPI, distribute_slice
from pyoperators.utils.testing import (assert_eq, assert_is, assert_is_not,
         assert_is_none, assert_not_in, assert_is_instance, assert_raises,
         assert_raises_if)

all_ops = [ eval('pyoperators.' + op) for op in dir(pyoperators) if op.endswith(
            'Operator')]

np.seterr(all='raise')

memory.verbose = True

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
        assert not getattr(operator.flags, f),'Operator {0} is {1}.'.format(
            operator, f) + (' ' + msg if msg else '')

def assert_is_inttuple(shape, msg=''):
    msg = '{0} is not an int tuple.'.format(shape) + (' ' + msg if msg else '')
    assert type(shape) is tuple, msg
    assert all([isinstance(s, int) for s in shape]), msg

def assert_square(op, msg=''):
    assert_flags(op, 'square', msg)
    assert_eq(op.shapein, op.shapeout)
    assert_eq(op.reshapein, op.reshapeout)
    assert_eq(op.toshapein, op.toshapeout)

dtypes = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64,
          np.float128, np.complex64, np.complex128, np.complex256)]

shapes = (None, (), (1,), (2,3))

class ndarray2(np.ndarray):
    pass

class ndarray3(np.ndarray):
    pass

class ndarray4(np.ndarray):
    pass

@decorators.square
class Op2(Operator):
    attrout = {'newattr':True}
    def direct(self, input, output):
        pass
    def transpose(self, input, output):
        pass            

@decorators.square
class Op3(Operator):
    classout = ndarray3
    classin = ndarray4
    def direct(self, input, output):
        pass
    def transpose(self, input, output):
        pass

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

@decorators.real
@decorators.symmetric
class HomothetyOutplaceOperator(Operator):
    def __init__(self, value, **keywords):
        Operator.__init__(self, **keywords)
        self.value = value
    def direct(self, input, output):
        output[...] = self.value * input


#===========
# Test flags
#===========

def test_flags():
    def func(o):
        if o.flags.idempotent:
            assert_is(o, o * o)
        if o.flags.real:
            assert_is(o, o.C)
        if o.flags.symmetric:
            assert_is(o, o.T)
        if o.flags.hermitian:
            assert_is(o, o.H)
        if o.flags.involutary:
            assert_is(o, o.I)
        if o.flags.orthogonal:
            assert_is(o.T, o.I)
        if o.flags.unitary:
            assert_is(o.H, o.I)
    v = np.arange(10.)
    for op in all_ops:
        try:
            o = op()
        except:
            try:
                o = op(v)
            except:
                print 'Cannot test: ' + op.__name__
                continue
        if type(o) is not op:
            print 'Cannot test: ' + op.__name__
            continue
        yield func, o

def test_symmetric():    
    mat = np.matrix([[2,1],[1,2]])
    @decorators.symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=2, dtype=mat.dtype)
        def direct(self, input, output):
            output[...] = np.dot(mat, input)

    op = Op()
    assert_flags(op, 'linear,square,real,symmetric')
    assert_eq(op.shape, (2,2))
    assert_eq(op.shapeout, (2,))
    assert op is op.C
    assert op is op.T
    assert op is op.H
    assert_eq(op([1,1]), np.array(mat * [[1],[1]]).ravel())


#========================
# Test input/output shape
#========================

def test_shape_is_inttuple():
    def func(o):
        shapein = o.shapein
        assert_is_inttuple(o.shapein)
        assert_is_inttuple(o.shapeout)
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3,2], np.array([3,2]), (3,2),
                    [3.,2], np.array([3.,2]), (3.,2)):
        o = Operator(shapein=shapein, shapeout=shapein)
        yield func, o

def test_shape_explicit():
    o1, o2, o3 = (
        Operator(shapeout=(13,2), shapein=(2,2)),
        Operator(shapeout=(2,2), shapein=(1,3)),
        Operator(shapeout=(1,3), shapein=4))
    def func(o, eout, ein):
        assert_eq(o.shapeout, eout)
        assert_eq(o.shapein, ein)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((13,2),(2,2),(13,2)),
                            ((1,3),(4,),(4,))):
        yield func, o, eout, ein
    assert_raises(ValueError, CompositionOperator, [o2, o1])
    assert_raises(ValueError, CompositionOperator, [o3, o2])
    assert_raises(ValueError, CompositionOperator, [o3, I, o1])

    o4 = Operator(shapeout=o1.shapeout)
    o5 = Operator(flags='square')

    o1 = Operator(shapein=(13,2), flags='square')
    for o in [o1+I, I+o1, o1+o4, o1+I+o5+o4, I+o5+o1]:
        yield func, o, o1.shapeout, o1.shapein
    assert_raises(ValueError, AdditionOperator, [o2, o1])
    assert_raises(ValueError, AdditionOperator, [o3, o2])
    assert_raises(ValueError, AdditionOperator, [I, o3, o1])
    assert_raises(ValueError, AdditionOperator, [o3, I, o1])

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
    def func(o, eout, ein):
        assert_eq(o.validatereshapein(shapein), eout)
        assert_eq(o.validatereshapeout(shapeout), ein)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((6,),(12,),(24,)),
                            ((4,),(2,),(1,))):
        yield func, o, eout, ein

def test_shapeout_unconstrained1():
    for shape in shapes:
        op = Operator(shapein=shape)
        assert_is_none(op.shapeout)

def test_shapeout_unconstrained2():
    class Op(Operator):
        def direct(self, input, output):
            output[...] = 4
    def func(s1, s2):
        op = IdentityOperator(shapein=s1) * Op(shapein=s2)
        if s1 is not None:
            assert op.shapeout == s1
        else:
            assert op.shapeout is None
    for s1 in shapes:
        for s2 in shapes:
            yield func, s1, s2

def test_shapeout_implicit():
    class Op(Operator):
        def reshapein(self, shape):
            return shape + (2,)
    def func(op, shapein):
        assert_flags_false(op, 'square')
        assert op.shapein == shapein
        if shapein is None:
            assert op.shapeout is None
        else:
            assert op.shapeout == shapein + (2,)
    for shapein in shapes:
        op = Op(shapein=shapein)
        yield func, op, shapein
    assert_raises(ValueError, Op, shapein=3, shapeout=11)

def test_shapein_unconstrained1():
    def func(shape):
        op = Operator(shapeout=shape)
        assert_is_none(op.shapein)
    for shape in shapes[1:]:
        yield func, shape

def test_shapein_unconstrained2():
    class Op(Operator):
        def reshapeout(self, shape):
            return shape + (2,)
    def func(op, shapeout):
        assert_flags_false(op, 'square')
        assert op.shapeout == shapeout
        assert op.shapein == shapeout + (2,)
    for shape in shapes[1:]:
        op = Op(shapeout=shape)
        yield func, op, shape
    assert_raises(ValueError, Op, shapein=3, shapeout=11)

def test_shapein_unconstrained3():
    @decorators.square
    class Op1(Operator):
        pass
    @decorators.square
    class Op2(Operator):
        def reshapein(self, shape):
            return shape
        def toshapein(self, v):
            return v
    @decorators.square
    class Op3(Operator):
        def reshapeout(self, shape):
            return shape
        def toshapeout(self, v):
            return v
    @decorators.square
    class Op4(Operator):
        def reshapein(self, shape):
            return shape
        def reshapeout(self, shape):
            return shape
        def toshapein(self, v):
            return v
        def toshapeout(self, v):
            return v

    def func(op, shape):
        assert_square(op)
        assert_eq(op.shapein, shape)
    for shape in shapes[1:]:
        for cls in (Op1, Op2, Op3, Op4):
            op = cls(shapeout=shape)
            yield func, op, shape


#================
# Test validation
#================

def test_validation():
    from .test_shared import Ops
    class ValidationError(ValueError):
        pass
    def vin(shape):
        if shape[0] % 2 == 0:
            raise ValidationError()
    def vout(shape):
        if shape[0] % 2 == 1:
            raise ValidationError()
    x_ok = np.empty(3)
    y_ok = np.empty(4)
    x_err = np.empty(6)
    y_err = np.empty(7)
    def func(cls):
        op = cls(validatein=vin, validateout=vout)
        op(x_ok, y_ok)
        cls_error = ValueError if op.flags.shape_input == 'explicit' else \
                    ValidationError
        assert_raises(cls_error, op, x_err, y_ok)
        cls_error = ValueError if op.flags.shape_output == 'explicit' else \
                    ValidationError
        assert_raises(cls_error, op, x_ok, y_err)

        if op.flags.shape_output == 'implicit':
            assert_raises(ValidationError, cls, validateout=vout,
                          shapein=x_err.shape)
        if op.flags.shape_input == 'implicit':
            assert_raises(ValidationError, cls, validatein=vin,
                          shapeout=y_err.shape)
    for cls in Ops:
        yield func, cls


#====================
# Test operator dtype
#====================

def test_dtype1():
    value = 2.5
    @decorators.square
    class Op(Operator):
        def __init__(self, dtype):
            Operator.__init__(self, dtype=dtype)
        def direct(self, input, output):
            np.multiply(input, np.array(value, self.dtype), output)
    input = complex(1,1)
    def func(dop, di):
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = Op(dop)(i)
        assert_eq(o.dtype, (i*np.array(value,dop)).dtype, str((dop,di)))
        assert_eq(o, i*np.array(value,dop), str((dop,di)))

    for dop in dtypes:
        for di in dtypes:
            yield func, dop, di

def test_dtype2():
    @decorators.square
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)
    op = Op()
    input = complex(1,1)
    def func(di):
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = op(i)
        assert_eq(o.dtype, (i * i).dtype, str(di))
        assert_eq(o, i * i, str(di))
    for di in dtypes:
        yield func, di

#=========================
# Test operator comparison
#=========================

def test_eq():
    from .test_shared import ops as ops1
    ops2 = [type(o)() for o in ops1]
    def func(op1, op2):
        assert_eq(op1, op2)
    for op1, op2 in zip(ops1, ops2):
        yield func, op1, op2


#================
# Test iadd, imul
#================

def test_iadd_imul():
    from .test_shared import ops
    def func(op1, op2, operation):
        if operation is operator.iadd:
            op = op1 + op2
            op1 += op2
        else:
            op = op1 * op2.T
            op1 *= op2.T
        assert_eq(op1, op)
    for operation in (operator.iadd, operator.imul):
        for op2 in ops:
            for op1 in ops:
                yield func, op1, op2, operation

#===========================
# Test attribute propagation
#===========================

def test_propagation_attribute1():
    @decorators.square
    class AddAttribute(Operator):
        attrout = {'newattr_direct':True}
        attrin = {'newattr_transpose':True}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    @decorators.square
    class AddAttribute2(Operator):
        attrout = {'newattr_direct':False}
        attrin = {'newattr_transpose':False}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    @decorators.square
    class AddAttribute3(Operator):
        attrout = {'newattr3_direct':True}
        attrin = {'newattr3_transpose':True}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    inputs = [np.ones(5), np.ones(5).view(ndarray2)]
    def func1(i):
        op = AddAttribute()
        assert op(i).newattr_direct
        assert op.T(i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        assert not op(i).newattr_direct
        assert op.T(i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        assert op(i).newattr_direct
        assert op(i).newattr3_direct
        assert op.T(i).newattr_transpose
        assert op.T(i).newattr3_transpose
    for i in inputs:
        yield func1, i

    def func2(i_):
        op = AddAttribute()
        i = i_.copy()
        assert op(i,i).newattr_direct
        i = i_.copy()
        assert op.T(i,i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        i = i_.copy()
        assert not op(i,i).newattr_direct
        i = i_.copy()
        assert op.T(i,i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        i = i_.copy()
        o = op(i,i)
        assert o.newattr_direct
        assert o.newattr3_direct
        i = i_.copy()
        o = op.T(i,i)
        assert o.newattr_transpose
        assert o.newattr3_transpose
    for i_ in inputs:
        yield func2, i_


def test_propagation_attribute2():
    @decorators.square
    class Op(Operator):
        attrin = {'attr_class':1, 'attr_instance':2, 'attr_other':3}
        attrout = {'attr_class':4, 'attr_instance':5, 'attr_other':6}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass
    class ndarray2(np.ndarray):
        attr_class = 10
        def __new__(cls, data):
            result = np.ndarray(data).view(cls)
            result.attr_instance = 11
            return result

    op = Op()
    output = op(ndarray2(1))
    assert output.__dict__ == op.attrout
    output = op.T(ndarray2(1))
    assert output.__dict__ == op.attrin


def test_propagation_attribute3():
    class ndarraybase(np.ndarray):
        attr_class = None
        def __new__(cls, data):
            result = np.array(data).view(cls)
            return result
        def __array_finalize__(self, array):
            self.attr_class = 0
            self.attr_instance = 10
    class ndarray1(ndarraybase):
        attr_class1 = None
        def __new__(cls, data):
            result = ndarraybase(data).view(cls)
            return result
        def __array_finalize__(self, array):
            ndarraybase.__array_finalize__(self, array)
            self.attr_class1 = 1
            self.attr_instance1 = 11
    class ndarray2(ndarraybase):
        attr_class2 = None
        def __new__(cls, data):
            result = ndarraybase(data).view(cls)
            return result
        def __array_finalize__(self, array):
            ndarraybase.__array_finalize__(self, array)
            self.attr_class2 = 2
            self.attr_instance2 = 12
    @decorators.square
    class Op(Operator):
        classin = ndarray1
        classout = ndarray2
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    op = Op()
    input = ndarray1(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {'attr_instance':10, 'attr_instance1':11,
        'attr_instance2':12, 'attr_class':30, 'attr_class2':2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':42,
                               'attr_class':30, 'attr_class2':32}

    op = Op().T
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':41,
                               'attr_class':30, 'attr_class1':31}
    input = ndarray2(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {'attr_instance':10, 'attr_instance2':12,
        'attr_instance1':11, 'attr_class':30, 'attr_class1':1}

    op = Op().T * Op() # -> ndarray2 -> ndarray1
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':41,
                               'attr_class':30, 'attr_class1':1}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':11,
        'attr_instance2':42, 'attr_class':30, 'attr_class1':1}

    op = Op() * Op().T # -> ndarray1 -> ndarray2
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':12,
        'attr_instance1':41, 'attr_class':30, 'attr_class2':2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':42,
                               'attr_class':30, 'attr_class2':2}    

    
#=======================
# Test class propagation
#=======================

def check_propagation_class(op, i, c):
    o = op(i)
    assert_is(type(o), c)

def check_propagation_class_inplace(op, i, c):
    i = i.copy()
    op(i,i)
    assert_is(type(i), c)

def test_propagation_class():
    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    results = [[np.ndarray, ndarray2],
               [ndarraywrap, ndarray2],
               [ndarray3, ndarray3],
               [ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op, i, c

def test_propagation_class_inplace():
    inputs = [np.ones(2), np.ones(2).view(ndarray2), np.ones(2).view(ndarray3)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    results = [[np.ndarray, ndarray2, ndarray3],
               [np.ndarray, ndarray2, ndarray3],
               [np.ndarray, ndarray3, ndarray3],
               [np.ndarray, ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op, i, c

def test_propagation_classT():
    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    resultsT = [[np.ndarray, ndarray2],
                [np.ndarray, ndarray2],
                [ndarray4, ndarray4],
                [ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op.T, i, c

def test_propagation_classT_inplace():
    inputs = [np.ones(2), np.ones(2).view(ndarray2), np.ones(2).view(ndarray4)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    resultsT = [[np.ndarray, ndarray2, ndarray4],
                [np.ndarray, ndarray2, ndarray4],
                [np.ndarray, ndarray4, ndarray4],
                [np.ndarray, ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op.T, i, c

def test_propagation_class_nested():
    @decorators.square
    class O1(Operator):
        classout = ndarray2
        def direct(self, input, output):
            output[...] = input
    @decorators.square
    class O2(Operator):
        def direct(self, input, output):
            output[...] = input

    def func2(op1, op2, expected):
        assert_is((op1*op2)(1).__class__, expected)

    def func3(op1, op2, op3, expected):
        assert_is((op1*op2*op3)(1).__class__, expected)

    o1 = O1()
    o2 = O2()
    ops1 = [I, 2*I, o2, 2*I+o2]
    ops2 = [I+o1, 2*o1, o1+o2, o2+o1, I+2*o1, I+o1+o2, I+o2+o1, o1+I+o2,
            o1+o2+I, o2+o1+I, o2+I+o1]
    for op1 in ops1:
        for op2 in ops2:
            yield func2, op1, op2, ndarray2
            yield func2, op2, op1, ndarray2
    for op1 in ops1:
        for op2 in ops2:
            for op3 in ops1:
                yield func3, op1, op2, op3, ndarray2


#========================
# Test MPI communicators
#========================

def test_comm_commutative():
    comms_all = (None, MPI.COMM_SELF, MPI.COMM_WORLD)
    def func(operation, comms):
        ops = [Operator(commin=c) for c in comms]
        assert_raises_if(MPI.COMM_SELF in comms and MPI.COMM_WORLD in comms,
                         ValueError, operation, ops)
        ops = [Operator(commout=c) for c in comms]
        assert_raises_if(MPI.COMM_SELF in comms and MPI.COMM_WORLD in comms,
                         ValueError, operation, ops)
    for operation in (AdditionOperator, MultiplicationOperator):
        for comms in itertools.combinations_with_replacement(comms_all, 3):
            yield func, operation, comms

def test_comm_composition():
    comms_all = (None, MPI.COMM_SELF, MPI.COMM_WORLD)
    def func(commin, commout):
        ops = [Operator(commin=commin), Operator(commout=commout)]
        assert_raises_if(None not in (commin, commout) and commin is not \
                         commout, ValueError, CompositionOperator, ops)
    for commin, commout in itertools.product(comms_all, repeat=2):
        yield func, commin, commout


#===========================
# Test in-place/out-of-place
#===========================

def test_inplace1():
    memory.stack = []
    @decorators.square
    class NotInplace(Operator):
        def direct(self, input, output):
            output[...] = 0
            output[0] = input[0]
    op = NotInplace()
    v = np.array([2., 0., 1.])
    op(v,v)
    assert_eq(v,[2,0,0])
    assert_eq(len(memory.stack), 1)


def test_inplace_can_use_output():

    A = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0] : 'A',
           B.__array_interface__['data'][0] : 'B',
           C.__array_interface__['data'][0] : 'C',
           D.__array_interface__['data'][0] : 'D',
           }

    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace':inplace})
            self.log = log
        def direct(self, input, output):
            if self.flags.inplace:
                tmp = input[0]
                output[1:] = 2 * input
                output[0] = tmp
            else:
                output[...] = 0
                output[0] = input[0]
                output[1:] = 2 * input
            try:
                self.log.insert(0, ids[output.__array_interface__['data'][0]])
            except KeyError:
                self.log.insert(0, 'unknown')
        def reshapein(self, shape):
            return (shape[0]+1,)

    def show_stack():
        return ''.join([ids[s.__array_interface__['data'][0]] \
                            for s in memory.stack])

    expecteds_outplace = {
        2 : ['BBA',   #II
             'BBA',   #IO
             'BCA',   #OI
             'BCA'],  #OO
        3 : ['BBBA',  #III
             'BBBA',  #IIO
             'BBCA',  #IOI
             'BBCA',  #IOO
             'BCCA',  #OII
             'BCCA',  #OIO
             'BCBA',  #OOI
             'BCBA'], #OOO
        4 : ['BBBBA', #IIII
             'BBBBA', #IIIO
             'BBBCA', #IIOI
             'BBBCA', #IIOO
             'BBCCA', #IOII
             'BBCCA', #IOIO
             'BBCBA', #IOOI
             'BBCBA', #IOOO
             'BCCCA', #OIII
             'BCCCA', #OIIO
             'BCCBA', #OIOI
             'BCCBA', #OIOO
             'BCBBA', #OOII
             'BCBBA', #OOIO
             'BCBCA', #OOOI
             'BCBCA']}#OOOO

    expecteds_inplace = {
        2 : ['AAA',   #II
             'ABA',   #IO
             'ABA',   #OI
             'ABA'],  #OO
        3 : ['AAAA',  #III
             'ABBA',  #IIO
             'ABAA',  #IOI
             'AABA',  #IOO
             'ABAA',  #OII
             'ABBA',  #OIO
             'ABAA',  #OOI
             'ACBA'], #OOO
        4 : ['AAAAA', #IIII
             'ABBBA', #IIIO
             'ABBAA', #IIOI
             'AAABA', #IIOO
             'ABAAA', #IOII
             'AABBA', #IOIO
             'AABAA', #IOOI
             'ABABA', #IOOO
             'ABAAA', #OIII
             'ABBBA', #OIIO
             'ABBAA', #OIOI
             'ABABA', #OIOO
             'ABAAA', #OOII
             'ABABA', #OOIO
             'ABABA', #OOOI
             'ABABA']}#OOOO

    def func_outplace(n, i, expected, strops):
        memory.stack = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:8].view(float); v[0] = 1
        w = B[:(n+1)*8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_stack(), 'CD', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    def func_inplace(n, i, expected, strops):
        memory.stack = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:8].view(float); v[0] = 1
        w = A[:(n+1)*8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_stack(), 'BC', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, n, i, expected, strops

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, n, i, expected, strops

def test_inplace_cannot_use_output():

    A = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0] : 'A',
           B.__array_interface__['data'][0] : 'B',
           C.__array_interface__['data'][0] : 'C',
           D.__array_interface__['data'][0] : 'D',
           }
    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace':inplace})
            self.log = log
        def direct(self, input, output):
            if not self.flags.inplace:
                output[...] = 0
            output[:] = input[1:]
            try:
                self.log.insert(0, ids[output.__array_interface__['data'][0]])
            except KeyError:
                self.log.insert(0, 'unknown')
        def reshapein(self, shape):
            return (shape[0]-1,)

    def show_stack():
        return ''.join([ids[s.__array_interface__['data'][0]] \
                            for s in memory.stack])

    expecteds_outplace = {
        2 : ['BCA',   #II
             'BCA',   #IO
             'BCA',   #OI
             'BCA'],  #OO
        3 : ['BCCA',  #III
             'BCCA',  #IIO
             'BDCA',  #IOI
             'BDCA',  #IOO
             'BCCA',  #OII
             'BCCA',  #OIO
             'BDCA',  #OOI
             'BDCA'], #OOO
        4 : ['BCCCA', #IIII
             'BCCCA', #IIIO
             'BDDCA', #IIOI
             'BDDCA', #IIOO
             'BDCCA', #IOII
             'BDCCA', #IOIO
             'BCDCA', #IOOI
             'BCDCA', #IOOO
             'BCCCA', #OIII
             'BCCCA', #OIIO
             'BDDCA', #OIOI
             'BDDCA', #OIOO
             'BDCCA', #OOII
             'BDCCA', #OOIO
             'BCDCA', #OOOI
             'BCDCA']}#OOOO

    expecteds_inplace = {
        2 : ['ABA',   #II
             'ABA',   #IO
             'ABA',   #OI
             'ABA'],  #OO
        3 : ['ABBA',  #III
             'ABBA',  #IIO
             'ACBA',  #IOI
             'ACBA',  #IOO
             'ABBA',  #OII
             'ABBA',  #OIO
             'ACBA',  #OOI
             'ACBA'], #OOO
        4 : ['ABBBA', #IIII
             'ABBBA', #IIIO
             'ACCBA', #IIOI
             'ACCBA', #IIOO
             'ACBBA', #IOII
             'ACBBA', #IOIO
             'ABCBA', #IOOI
             'ABCBA', #IOOO
             'ABBBA', #OIII
             'ABBBA', #OIIO
             'ACCBA', #OIOI
             'ACCBA', #OIOO
             'ACBBA', #OOII
             'ACBBA', #OOIO
             'ABCBA', #OOOI
             'ABCBA']}#OOOO

    def func_outplace(n, i, expected, strops):
        memory.stack = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float); v[:] = range(n+1)
        w = B[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_stack(), 'CD', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    def func_inplace(n, i, expected, strops):
        memory.stack = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float); v[:] = range(n+1)
        w = A[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_stack(), 'BC', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, n, i, expected, strops

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, n, i, expected, strops

    
#===============
# Test addition
#===============

def test_addition():
    @decorators.square
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    memory.stack = []
    op = np.sum([Op(v) for v in [1]])
    assert_is(op.__class__, Op)

    op = np.sum([Op(v) for v in [1,2]])
    assert_eq(op.__class__, AdditionOperator)
    assert_eq(op(1), 3)
    assert_eq(len(memory.stack), 1)

    op = np.sum([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, AdditionOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    assert_eq(op(input, output), 7)
    assert_eq(input, 1)
    assert_eq(output, 7)
    assert_eq(len(memory.stack), 1)

    output = input
    assert_eq(op(input, output), 7)
    assert_eq(input, 7)
    assert_eq(output, 7)
    assert_eq(len(memory.stack), 2)


#=====================
# Test multiplication
#=====================

def test_multiplication():
    @decorators.square
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    memory.stack = []
    op = MultiplicationOperator([Op(v) for v in [1]])
    assert_is(op.__class__, Op)

    op = MultiplicationOperator([Op(v) for v in [1,2]])
    assert_eq(op.__class__, MultiplicationOperator)
    assert_eq(op(1), 2)
    assert_eq(len(memory.stack), 1)

    op = MultiplicationOperator([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, MultiplicationOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 1)

    output = input
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 2)


#==================
# Test composition
#==================

def test_composition1():
    def func(op, shapein, shapeout):
        assert_eq(op.shapein, shapein)
        assert_eq(op.shapeout, shapeout)
        if shapein is not None and shapein == shapeout:
            assert_flags(op, 'square')
    for shapein in shapes:
        for shapemid in shapes:
            if shapemid is None and shapein is not None:
                continue
            op1 = Operator(shapein=shapein, shapeout=shapemid)
            for shapeout in shapes:
                if shapeout is None and shapemid is not None:
                    continue
                op2 = Operator(shapein=shapemid, shapeout=shapeout)
                op = op2 * op1
                yield func, op, shapein, shapeout

def test_composition2():
    class Op(Operator):
        def reshapein(self, shapein):
            return 2*shapein

    def func(op, shape):
        assert op.shapein is None
        assert op.shapeout == (2*shape if shape is not None else None)
        assert_flags_false(op, 'square')
    for shape in shapes:
        op = Op() * Operator(shapeout=shape)
        yield func, op, shape

    op = Op() * Op()
    assert op.shapein is None
    assert op.shapeout is None
    assert_flags_false(op, 'square')

def test_composition3():
    @decorators.square
    @decorators.inplace
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    memory.stack = []
    op = np.product([Op(v) for v in [1]])
    assert_is(op.__class__, Op)
    op(1)
    assert_eq(len(memory.stack), 0)

    memory.stack = []
    op = np.product([Op(v) for v in [1,2]])
    assert_is(op.__class__, CompositionOperator)
    assert_eq(op(1), 2)
    assert_eq(len(memory.stack), 0)

    memory.stack = []
    assert_eq(op([1]), 2)
    assert_eq(len(memory.stack), 0)

    op = np.product([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, CompositionOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    memory.stack = []
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 0)

    output = input
    memory.stack = []
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 0)

    input = np.array([1], int)
    output = np.array([0], int)
    memory.stack = []
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 0)

    output = input
    memory.stack = []
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(output, 8)
    assert_eq(len(memory.stack), 0)


#================
# Test partition
#================

def test_partition1():
    o1 = HomothetyOperator(1, shapein=1)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=3)
    
    r = DiagonalOperator([1,2,2,3,3,3]).todense()
    def func(ops, p):
        op = BlockDiagonalOperator(ops, partitionin=p, axisin=0)
        assert_eq(op.todense(6), r, str(op))
    for ops, p in zip(((o1,o2,o3), (I,o2,o3), (o1,2*I,o3), (o1,o2,3*I)),
                      (None, (1,2,3), (1,2,3), (1,2,3))):
        yield func, ops, p

def test_partition2():
    # in some cases in this test, partitionout cannot be inferred from
    # partitionin, because the former depends on the input rank
    i = np.arange(3*4*5*6).reshape(3,4,5,6)
    def func(axisp, p, axiss):
        op = BlockDiagonalOperator(3*[Stretch(axiss)], partitionin=p,
                                   axisin=axisp)
        assert_eq(op(i), Stretch(axiss)(i))
    for axisp,p in zip((0,1,2,3,-1,-2,-3), ((1,1,1),(1,2,1),(2,2,1),(2,3,1),
                                            (2,3,1),(2,2,1),(1,2,1),(1,1,1))):
        for axiss in (0,1,2,3):
            yield func, axisp, p, axiss

def test_partition3():
    # test axisin != axisout...
    pass

def test_partition4():
    o1 = HomothetyOperator(1, shapein=1)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=3)
    @decorators.universal
    class Op(Operator):
        pass
    op=Op()
    p=BlockDiagonalOperator([o1,o2,o3], axisin=0)
    
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)


def test_block1():
    ops = [HomothetyOperator(i, shapein=(2,2)) for i in range(1,4)]
    def func(axis, s):
        op = BlockDiagonalOperator(ops, new_axisin=axis)
        assert_eq(op.shapein, s)
        assert_eq(op.shapeout, s)
    for axis,s in zip(range(-3,3),
                      ((3,2,2),(2,3,2),(2,2,3),(3,2,2),(2,3,2),(2,2,3))):
        yield func, axis, s
def test_block2():
    shape = (3,4,5,6)
    i = np.arange(np.product(shape)).reshape(shape)
    def func(axisp, axiss):
        op = BlockDiagonalOperator(shape[axisp]*[Stretch(axiss)], new_axisin=axisp)
        axisp_ = axisp if axisp >= 0 else axisp + 4
        axiss_ = axiss if axisp_ > axiss else axiss + 1
        assert_eq(op(i), Stretch(axiss_)(i))
    for axisp in (0,1,2,3,-1,-2,-3):
        for axiss in (0,1,2):
            yield func, axisp, axiss

def test_block3():
    # test new_axisin != new_axisout...
    pass

def test_block4():
    o1 = HomothetyOperator(1, shapein=2)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=2)
    @decorators.universal
    class Op(Operator):
        pass
    op=Op()
    p=BlockDiagonalOperator([o1,o2,o3], new_axisin=0)
    
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)

def test_block_column1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, BlockColumnOperator, [I2, 2*I3], axisout=0)
    assert_raises(ValueError, BlockColumnOperator, [I2, 2*I3], new_axisout=0)

def test_block_column2():
    p = np.matrix([[1,0], [0,2], [1,0]])
    o = asoperator(np.matrix(p))
    e = BlockColumnOperator([o, 2*o], axisout=0)
    assert_eq(e.todense(), np.vstack([p,2*p]))
    assert_eq(e.T.todense(), e.todense().T)
    e = BlockColumnOperator([o, 2*o], new_axisout=0)
    assert_eq(e.todense(), np.vstack([p,2*p]))
    assert_eq(e.T.todense(), e.todense().T)

def test_block_row1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, BlockRowOperator, [I2, 2*I3], axisin=0)
    assert_raises(ValueError, BlockRowOperator, [I2, 2*I3], new_axisin=0)

def test_block_row2():
    p = np.matrix([[1,0], [0,2], [1,0]])
    o = asoperator(np.matrix(p))
    r = BlockRowOperator([o, 2*o], axisin=0)
    assert_eq(r.todense(), np.hstack([p,2*p]))
    assert_eq(r.T.todense(), r.todense().T)
    r = BlockRowOperator([o, 2*o], new_axisin=0)
    assert_eq(r.todense(), np.hstack([p,2*p]))
    assert_eq(r.T.todense(), r.todense().T)

def test_partition_implicit_commutative():
    partitions = (None,None), (2,None), (None,3), (2,3)
    ops = [I, 2*I]
    def func(op1, op2, p1, p2, cls):
        op = operation([op1, op2])
        assert type(op) is cls
        if op.partitionin is None:
            assert op1.partitionin is op2.partitionin is None
        else:
            assert op.partitionin == merge_none(p1,p2)
        if op.partitionout is None:
            assert op1.partitionout is op2.partitionout is None
        else:
            assert op.partitionout == merge_none(p1,p2)
    for operation in (AdditionOperator, MultiplicationOperator):
        for p1 in partitions:
            for p2 in partitions:
                for cls, aout, ain, pout1, pin1, pout2, pin2 in zip(
                    (BlockRowOperator, BlockDiagonalOperator,
                     BlockColumnOperator), (None,0,0), (0,0,None), (None,p1,p1),
                    (p1,p1,None), (None,p2,p2), (p2,p2,None)):
                    op1 = BlockOperator(ops, partitionout=pout1,
                              partitionin=pin1, axisin=ain, axisout=aout)
                    op2 = BlockOperator(ops, partitionout=pout2,
                              partitionin=pin2, axisin=ain, axisout=aout)
                    yield func, op1, op2, p1, p2, cls

def test_partition_implicit_composition():
    partitions = (None,None), (2,None), (None,3), (2,3)
    ops = [I, 2*I]
    def func(op1, op2, pin1, pout2, cls):
        op = op1 * op2
        assert_is_instance(op, cls)
        if not isinstance(op, BlockOperator):
            return
        pout = None if isinstance(op, BlockRowOperator) else \
               merge_none(pin1, pout2)
        pin = None if isinstance(op, BlockColumnOperator) else \
              merge_none(pin1, pout2)
        assert pout == op.partitionout
        assert pin == op.partitionin
    for pin1 in partitions:
        for pout2 in partitions:
            for cls1, cls2, cls, aout1, ain1, aout2, ain2, pout1, pin2, in zip(
                (BlockRowOperator, BlockRowOperator, BlockDiagonalOperator,
                BlockDiagonalOperator), (BlockDiagonalOperator,
                BlockColumnOperator, BlockDiagonalOperator,
                BlockColumnOperator), (BlockRowOperator, HomothetyOperator,
                BlockDiagonalOperator, BlockColumnOperator), (None,None,0,0),
                (0,0,0,0), (0,0,0,0), (0,None,0,None), (None,None,pin1,pin1),
                (pout2,None,pout2,None)):
                op1 = BlockOperator(ops, partitionin=pin1, partitionout=pout1,
                                    axisout=aout1, axisin=ain1)
                op2 = BlockOperator(ops, partitionout=pout2, partitionin=pin2,
                                    axisout=aout2, axisin=ain2)
                yield func, op1, op2, pin1, pout2, cls


def test_partition_broadcast_composition():
    def func1(d, b):
        p = d * b
        assert_is_instance(p, BlockDiagonalOperator)
        d_ = d.todense(b.shapeout)
        b_ = b.todense()
        p_ = np.dot(d_, b_)
        assert_eq(p.todense(), p_)
    def func2(d, b):
        p = d + b
        assert_is_instance(p, BlockDiagonalOperator)
        d_ = d.todense(b.shapein)
        b_ = b.todense()
        p_ = np.add(d_, b_)
        assert_eq(p.todense(), p_)
    def func3(b, d):
        p = b * d
        assert_is_instance(p, BlockDiagonalOperator)
        b_ = b.todense()
        d_ = d.todense(b.shapein)
        p_ = np.dot(b_, d_)
        assert_eq(p.todense(), p_)
        
    for ndims in range(4):
        shape = tuple(range(2,2+ndims))
        sfunc1 = lambda ndim: np.arange(np.product(range(2,ndim+2))).reshape(
                             range(2,ndim+2)) + 2
        sfunc2 = lambda ndim: np.arange(np.product(range(2+ndims-ndim,2+ndims))).reshape(
                             range(2+ndims-ndim,2+ndims)) + 2
        diag = [DiagonalOperator(sfunc1(ndim)) for ndim in range(ndims+1)] + \
               [DiagonalOperator(sfunc2(ndim), broadcast='leftward')
                    for ndim in range(1,ndims+1)] + \
               [DiagonalOperator(sfunc1(ndim), broadcast='rightward')
                    for ndim in range(1,ndims+1)]

        def toone(index):
            list_ = list(shape)
            list_[index] = 1
            return list_
        def remove(index):
            list_ = list(shape)
            list_.pop(index)
            return list_
        block = [BlockDiagonalOperator([HomothetyOutplaceOperator(v, shapein=toone(axis)) for v in range(2, 2+shape[axis])], axisin=axis, partitionin=shape[axis]*[1]) for axis in range(-ndims,ndims)] + \
                [BlockDiagonalOperator([HomothetyOutplaceOperator(v, shapein=remove(axis)) for v in range(2, 2+shape[axis])], new_axisin=axis, partitionin=shape[axis]*[1]) for axis in range(-ndims,ndims)]

        for d, b in itertools.product(diag, block):
            if d.broadcast == 'disabled' and d.shapein != b.shapeout:
                continue
            yield func1, d, b
            yield func2, d, b
            yield func3, b, d


#==================
# Test Block slice
#==================

def test_block_slice():
    size = 4
    def func(o, input, expected):
        actual = o(input)
        assert_eq(actual, expected)
        o(input, input)
        assert_eq(input, expected)
    for ndim in range(1,5):
        for nops in range(1,5):
            for Op in [HomothetyOperator, HomothetyOutplaceOperator]:
                slices_ = [
                    [distribute_slice(size, i, nops) for i in range(nops)],
                    [distribute_slice(size, i, size) for i in range(nops)],
                    [ndim * [slice(i, None, nops)] for i in range(nops)],
                ]
                for slices in slices_:
                    input = np.zeros(ndim*(size,))
                    expected = np.zeros_like(input)
                    ops = [Op(i+1) for i in range(nops)]
                    for i, s in enumerate(slices):
                        input[s] = 10 * (i+1)
                        expected[s] = input[s] * (i+1)
                    o = BlockSliceOperator(ops, slices)
                    assert o.flags.inplace is Op.flags.inplace
                    yield func, o, input, expected


#=============================
# Test non-composite operators
#=============================

def test_broadcasting_as_strided():
    shapes = {'leftward':(2,4,3,4,2,2), 'rightward':(3,2,2,3,1,2)}
    def func(b):
        o = BroadcastingOperator(np.arange(6).reshape((3,1,2,1)), broadcast=b)
        s = shapes[b]
        if b == 'leftward':
            v = o.data*np.ones(s)
        else:
            v = (o.data.T * np.ones(s, int).T).T
        assert_eq(o._as_strided(s), v)
    for b in ('rightward', 'leftward'):
        yield func, b

def test_diagonal1():
    data = (0., 1., [0,0], [1,1], 2, [2,2], [1,2])
    expected = (ZeroOperator, IdentityOperator, ZeroOperator, IdentityOperator,
                HomothetyOperator, HomothetyOperator, DiagonalOperator)
    def func(d, e):
        assert_is(DiagonalOperator(d).__class__, e)
    for d, e in zip(data, expected):
        yield func, d, e

def test_diagonal2():
    ops = (
           DiagonalOperator([1.,2], broadcast='rightward'),
           DiagonalOperator([[2.,3,4],[5,6,7]], broadcast='rightward'),
           DiagonalOperator([1.,2,3,4,5], broadcast='leftward'),
           DiagonalOperator(np.arange(20.).reshape(4,5), broadcast='leftward'),
           DiagonalOperator(np.arange(120.).reshape(2,3,4,5)),
           HomothetyOperator(7.),
           IdentityOperator(),
          )
    def func(op, d1, d2):
        d = op(d1, d2)
        if type(d1) is DiagonalOperator:
            assert_is(type(d), DiagonalOperator)
        elif type(d1) is HomothetyOperator:
            assert_is(type(d), HomothetyOperator)
        elif op is np.multiply:
            assert_is(type(d), IdentityOperator)
        else:
            assert_is(type(d), HomothetyOperator)

        data = op(d1.data.T, d2.data.T).T if 'rightward' in (d1.broadcast,
               d2.broadcast) else op(d1.data, d2.data)
        assert_eq(d.data, data)
    for op in (np.add, np.multiply):
        for i in range(7):
            d1 = ops[i]
            for j in range(i,7):
                d2 = ops[j]
                if set((d1.broadcast, d2.broadcast)) == set(('leftward', 'rightward')):
                    continue
                yield func, op, d1, d2
    
def test_homothety_operator():
    s = HomothetyOperator(1)
    assert s.C is s.T is s.H is s.I is s

    s = HomothetyOperator(-1)
    assert s.C is s.T is s.H is s.I is s

    s = HomothetyOperator(2.)
    assert s.C is s.T is s.H is s
    assert_is_not(s.I, s)
    def func(o):
        assert_is_instance(o, HomothetyOperator)
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield func, o

    s = HomothetyOperator(complex(1,1))
    assert_is(s.T, s)
    assert_is(s.H, s.C)
    assert_not_in(s.I, (s, s.C))
    assert_not_in(s.I.C, (s, s.C))
    assert_is_instance(s.C, HomothetyOperator)
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield func, o

def test_homothety_reduction1():
    models = 1.*I+I, -I, (-2) * I, -(2*I), 1.*I-I, 1.*I-2*I
    results = [6, -3, -6, -6, 0, -3]
    def func(model, result, i):
        o = model(i)
        assert_eq(o, result, str((model,i)))
        assert_eq(o.dtype, int, str((model,i)))
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield func, model, result, i

def test_homothety_reduction2():
    model = -I
    iops = (operator.iadd, operator.isub, operator.imul, operator.iadd,
            operator.imul)
    imodels = 2*I, 2*I, 2*I, O, O
    results = [3, -3, -6, -6, 0]
    def func(imodel, result, i):
        assert_eq(model(i), result)
    for iop, imodel, result in zip(iops, imodels, results):
        model = iop(model, imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield func, imodel, result, i

def test_homothety_reduction3():
    @decorators.linear
    class Op(Operator):
        pass
    def func(opout, opin, idin):
        if opin is not None and idin is not None and opin != idin:
            return
        p = Op(shapeout=opout, shapein=opin) * IdentityOperator(shapein=idin)

        if idin is None:
            idin = opin
        assert_is_instance(p, Op)
        assert_eq(p.shapein, idin)
        assert_eq(p.shapeout, opout)
    for opout in (None, (100,)):
        for opin in (None, (100,)):
            for idin in (None, (100,)):
                yield func, opout, opin, idin

def test_constant_reduction1():
    c = 1, np.array([1,2]), np.array([2,3,4])
    t = 'scalar', 'rightward', 'leftward'

    def func(c1, t1, c2, t2):
        op2 = ConstantOperator(c2, broadcast=t2)
        op = op1 + op2
        if set((op1.broadcast, op2.broadcast)) != set(('rightward', 'leftward')):
            assert_is_instance(op, ConstantOperator)
        v = np.zeros((2,3))
        op(np.nan, v)
        z = np.zeros((2,3))
        if t1 == 'rightward':
            z.T[...] += c1.T
        else:
            z[...] += c1
        if t2 == 'rightward':
            z.T[...] += c2.T
        else:
            z[...] += c2
        assert_eq(v, z)
    for c1, t1 in zip(c, t):
        op1 = ConstantOperator(c1, broadcast=t1)
        for c2, t2 in zip(c, t):
            yield func, c1, t1, c2, t2

def test_constant_reduction2():
    H = HomothetyOperator
    C = CompositionOperator
    D = DiagonalOperator
    cs = (ConstantOperator(3), ConstantOperator([1,2,3], broadcast='leftward'),
          ConstantOperator(np.ones((2,3))))
    os = (I, H(2, shapein=(2,3)) * Operator(direct=np.square, shapein=(2,3), 
          flags='square'), H(5))
    results = (((H, 3), (C, (H, 6)), (H, 15)),
               ((D, [1,2,3]), (C, (D, [2,4,6])), (D, [5,10,15])),
               ((IdentityOperator, 1), (C, (H,2)), (H, 5))
               )
    
    v = np.arange(6).reshape((2,3))
    def func(c, o, r):
        op = MultiplicationOperator([c, o])
        assert_eq(op(v), c.data*o(v))
        assert_is(type(op), r[0])
        if type(op) is CompositionOperator:
            op = op.operands[0]
            r = r[1]
            assert_is(type(op), r[0])
        assert_eq, op.data, r[1]
    for c,rs in zip(cs, results):
        for o, r in zip(os, rs):
            yield func, c, o, r

def _test_constant_reduction3():
    @decorators.square
    class Op(Operator):
        def direct(self, input, output):
            output[...] = input + np.arange(input.size).reshape(input.shape)
    os = (Op(shapein=()), Op(shapein=(4)), Op(shapein=(2,3,4)))
    cs = (ConstantOperator(2), ConstantOperator([2], broadcast='leftward'),
          ConstantOperator(2*np.arange(8).reshape((2,1,4)), broadcast='leftward'))
    v = 10000000
    def func(o, c):
        op = o * c
        y_tmp = np.empty(o.shapein, int)
        c(v, y_tmp)
        assert_eq(op(v), o(y_tmp))
    for o, c in zip(os, cs):
        yield func, o, c

def test_zero1():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6)
    zo = z*o
    assert_is_instance(zo, ZeroOperator)
    assert_eq(zo.shapein, o.shapein)
    assert_is_none(zo.shapeout)

def test_zero2():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator()
    zo = z*o
    assert_is_instance(zo, ZeroOperator)
    assert_is_none(zo.shapein, 'in')
    assert_eq(zo.shapeout, z.shapeout, 'out')

def test_zero3():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator(flags='square')
    zo = z*o
    assert_is_instance(zo, ZeroOperator)
    assert_eq(zo.shapein, z.shapein, 'in')
    assert_eq(zo.shapeout, z.shapeout, 'out')

def test_zero4():
    z = ZeroOperator()
    o = Operator(flags='linear')
    assert_is_instance(z*o, ZeroOperator)
    assert_is_instance(o*z, ZeroOperator)

def test_zero5():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6, flags='linear')
    zo = z*o
    oz = o*z
    assert_is_instance(zo, ZeroOperator, 'zo')
    assert_eq(zo.shapein, o.shapein, 'zo in')
    assert_is_none(zo.shapeout, 'zo out')
    assert_is_instance(oz, ZeroOperator, 'oz')
    assert_is_none(oz.shapein, 'oz, in')
    assert_eq(oz.shapeout, o.shapeout, 'oz, out')

def test_zero6():
    z = ZeroOperator(flags='square')
    @decorators.linear
    class Op(Operator):
        def direct(self, input, output):
            output[:] = np.concatenate([input, 2*input])
        def transpose(self, input, output):
            output[:] = input[0:output.size]
        def reshapein(self, shapein):
            return (2 * shapein[0],)
        def reshapeout(self, shapeout):
            return (shapeout[0] // 2,)
    o = Op()
    zo = z*o
    oz = o*z
    v = np.ones(4)
    assert_eq(zo.T(v), o.T(z.T(v)))
    assert_eq(oz.T(v), z.T(o.T(v)))
    
def test_zero7():
    z = ZeroOperator()
    assert_is(z*z, z)
