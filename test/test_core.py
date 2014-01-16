import itertools
import numpy as np
import operator
import sys

from nose.plugins.skip import SkipTest
from pyoperators import memory, decorators
from pyoperators.core import (
    Operator, AdditionOperator, BlockColumnOperator, BlockDiagonalOperator,
    BlockRowOperator, BlockSliceOperator, CompositionOperator, GroupOperator,
    ConstantOperator, DenseOperator, DiagonalOperator, IdentityOperator,
    MaskOperator, MultiplicationOperator, HomothetyOperator, ReductionOperator,
    ZeroOperator, asoperator, _pool as pool, I, O)
from pyoperators.memory import zeros
from pyoperators.utils import (
    ndarraywrap, first_is_not, operation_assignment, product)
from pyoperators.utils.mpi import MPI, distribute_slice
from pyoperators.utils.testing import (
    assert_eq, assert_is, assert_is_not, assert_is_none, assert_not_in,
    assert_is_instance, assert_raises, assert_same, skiptest)
from .common import OPS, ALL_OPS, DTYPES, HomothetyOutplaceOperator

PYTHON_26 = sys.version_info < (2, 7)
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
        assert not getattr(operator.flags, f), 'Operator {0} is {1}.'.format(
            operator, f) + (' ' + msg if msg else '')


def assert_is_inttuple(shape, msg=''):
    msg = '{0} is not an int tuple.'.format(shape) + (' ' + msg if msg else '')
    assert type(shape) is tuple, msg
    assert all([isinstance(s, int) for s in shape]), msg


def assert_square(op, msg=''):
    assert_flags(op, 'square', msg)
    assert_eq(op.shapein, op.shapeout)
    assert_eq(op.reshapein, op.reshapeout)
    assert_eq(op.validatein, op.validateout)
    if op.shapein is None:
        assert_eq(op.toshapein, op.toshapeout)

SHAPES = (None, (), (1,), (3,), (2, 3))


class ndarray2(np.ndarray):
    pass


class ndarray3(np.ndarray):
    pass


class ndarray4(np.ndarray):
    pass


@decorators.square
class SquareOp(Operator):

    def __init__(self, **keywords):
        Operator.__init__(self, **keywords)


@decorators.square
class Op2(Operator):
    attrout = {'newattr': True}

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


class OperatorNIR(Operator):
    def direct(self, input, output):
        output[...] = input


class OperatorIR(Operator):
    def direct(self, input, output, operation=operation_assignment):
        operation(output, input)


#===========
# Test flags
#===========

def test_flags():
    def func(op):
        try:
            o = op()
        except:
            try:
                v = np.arange(10.)
                o = op(v)
            except:
                print 'Cannot test: ' + op.__name__
                return
        if type(o) is not op:
            print 'Cannot test: ' + op.__name__
            return
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
    for op in ALL_OPS:
        yield func, op


def test_symmetric():
    mat = np.matrix([[2, 1], [1, 2]])

    @decorators.symmetric
    class Op(Operator):

        def __init__(self):
            Operator.__init__(self, shapein=2, dtype=mat.dtype)

        def direct(self, input, output):
            output[...] = np.dot(mat, input)

    op = Op()
    assert_flags(op, 'linear,square,real,symmetric')
    assert_eq(op.shape, (2, 2))
    assert_eq(op.shapeout, (2,))
    assert op is op.C
    assert op is op.T
    assert op is op.H
    assert_eq(op([1, 1]), np.array(mat * [[1], [1]]).ravel())


def test_shape_input_and_output():
    ops = tuple(cls() for cls in OPS)
    kind = {'Expl': 'explicit', 'Impl': 'implicit', 'Unco': 'unconstrained'}

    def func(flags, name):
        assert_eq(flags.shape_output, kind[name[:4]])
        assert_eq(flags.shape_input, kind[name[4:]])

    for op in ops:
        yield func, op.flags, type(op).__name__


def test_inplace_reduction1():
    out = np.zeros(3, dtype=int)
    assert not OperatorNIR().flags.inplace_reduction
    assert_raises(ValueError, OperatorNIR(), [1, 0, 0], out,
                  operation=operator.iadd)
    assert OperatorIR().flags.inplace_reduction
    assert_raises(ValueError, OperatorIR(), [1, 0, 0], operation=operator.iadd)


def test_inplace_reduction2():
    op = OperatorIR()
    inputs = [1, 1, 0], [0, 2, 1], [0, 1, 1]
    expecteds = [0, 1, 1], [2, 5, 3], [0, 2, 0]

    def func(o, e):
        output = np.ones(3, dtype=int)
        for i in inputs:
            op(i, output, operation=o)
        assert_same(output, e)
    for o, e in zip((operation_assignment, operator.iadd, operator.imul),
                    expecteds):
        yield func, o, e

def test_autoflags():
    def func(f):
        assert_raises(ValueError, Operator, flags=f)
    for f in ['shape_input', 'shape_output', 'inplace_reduction']:
        yield func, f


#==================
# Test conjugation
#==================

def test_conjugation():
    @decorators.square
    class OpBase(Operator):
        def __init__(self, data_=None):
            Operator.__init__(self, shapein=2, dtype=complex)
            if data_ is None:
                data_ = data
            self.data = data_
            self.dataI = np.linalg.inv(data_)

        def direct(self, input, output):
            np.dot(self.data, input, output)

        def inverse(self, input, output):
            np.dot(self.dataI, input, output)

    class Op1T(OpBase):
        def transpose(self, input, output):
            np.dot(self.data.T, input, output)

    class Op1H(OpBase):
        def adjoint(self, input, output):
            np.dot(self.data.T.conj(), input, output)

    class Op1IT(OpBase):
        def inverse_transpose(self, input, output):
            np.dot(self.dataI.T, input, output)

    class Op1IH(OpBase):
        def inverse_adjoint(self, input, output):
            np.dot(self.dataI.T.conj(), input, output)

    class Op2T(OpBase):
        def __init__(self):
            OpBase.__init__(self)
            self.set_rule('T', lambda s: OpBase(s.data.T))

    class Op2H(OpBase):
        def __init__(self):
            OpBase.__init__(self)
            self.set_rule('H', lambda s: OpBase(s.data.T.conj()))

    class Op2IT(OpBase):
        def __init__(self):
            OpBase.__init__(self)
            self.set_rule('IT', lambda s: OpBase(s.dataI.T))

    class Op2IH(OpBase):
        def __init__(self):
            OpBase.__init__(self)
            self.set_rule('IH', lambda s: OpBase(s.dataI.T.conj()))

    data = np.array([[1, 1j], [0, 2]])
    dense = OpBase().todense()
    denseI = np.linalg.inv(dense)

    def func(opT, opH, opIT, opIH):
        assert_eq(opT.C.todense(), dense.conj())
        assert_eq(opT.T.todense(), dense.T)
        assert_eq(opT.H.todense(), dense.T.conj())
        assert_eq(opH.C.todense(), dense.conj())
        assert_eq(opH.T.todense(), dense.T)
        assert_eq(opH.H.todense(), dense.T.conj())
        assert_eq(opIT.I.C.todense(), denseI.conj())
        assert_eq(opIT.I.T.todense(), denseI.T)
        assert_eq(opIT.I.H.todense(), denseI.T.conj())
        assert_eq(opIH.I.C.todense(), denseI.conj())
        assert_eq(opIH.I.T.todense(), denseI.T)
        assert_eq(opIH.I.H.todense(), denseI.T.conj())
    for opT, opH, opIT, opIH in [(Op1T(), Op1H(), Op1IT(), Op1IH()),
                                 (Op2T(), Op2H(), Op2IT(), Op2IH())]:
        yield func, opT, opH, opIT, opIH


#========================
# Test input/output shape
#========================

def test_shape_is_inttuple():
    def func(o):
        assert_is_inttuple(o.shapein)
        assert_is_inttuple(o.shapeout)
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3, 2], np.array([3, 2]), (3, 2),
                    [3., 2], np.array([3., 2]), (3., 2)):
        o = Operator(shapein=shapein, shapeout=shapein)
        yield func, o


def test_shape_explicit():
    o1, o2, o3 = (
        Operator(shapeout=(13, 2), shapein=(2, 2)),
        Operator(shapeout=(2, 2), shapein=(1, 3)),
        Operator(shapeout=(1, 3), shapein=4))

    def func(o, eout, ein):
        assert_eq(o.shapeout, eout)
        assert_eq(o.shapein, ein)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((13, 2), (2, 2), (13, 2)),
                            ((1, 3), (4,), (4,))):
        yield func, o, eout, ein
    assert_raises(ValueError, CompositionOperator, [o2, o1])
    assert_raises(ValueError, CompositionOperator, [o3, o2])
    assert_raises(ValueError, CompositionOperator, [o3, I, o1])

    o4 = Operator(shapeout=o1.shapeout)
    o5 = Operator(flags='square')

    o1 = Operator(shapein=(13, 2), flags='square')
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
            return super(Op, self).__str__() + 'x{0}'.format(self.factor)
    o1, o2, o3 = Op(2), Op(3), Op(4)
    assert o1.shapein is o2.shapein is o3.shapein is None
    shapein = (1,)
    shapeout = (24,)

    def func(o, eout, ein):
        assert_eq(o.reshapein(shapein), eout)
        assert_eq(o.reshapeout(shapeout), ein)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((6,), (12,), (24,)),
                            ((4,), (2,), (1,))):
        yield func, o, eout, ein


def test_shapeout_unconstrained1():
    for shape in SHAPES:
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
    for s1 in SHAPES:
        for s2 in SHAPES:
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
    for shapein in SHAPES:
        op = Op(shapein=shapein)
        yield func, op, shapein
    assert_raises(ValueError, Op, shapein=3, shapeout=11)


def test_shapein_unconstrained1():

    def func(shape):
        op = Operator(shapeout=shape)
        assert_is_none(op.shapein)
    for shape in SHAPES[1:]:
        yield func, shape


def test_shapein_unconstrained2():
    class Op(Operator):

        def reshapeout(self, shape):
            return shape + (2,)

    def func(op, shapeout):
        assert_flags_false(op, 'square')
        assert op.shapeout == shapeout
        assert op.shapein == shapeout + (2,)
    for shape in SHAPES[1:]:
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
    for shape in SHAPES[1:]:
        for cls in (Op1, Op2, Op3, Op4):
            op = cls(shapeout=shape)
            yield func, op, shape


#================
# Test validation
#================

def test_validation():
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
    for cls in OPS:
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
    input = complex(1, 1)

    def func(dop, di):
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = Op(dop)(i)
        assert_eq(o.dtype, (i*np.array(value, dop)).dtype, str((dop, di)))
        assert_eq(o, i*np.array(value, dop), str((dop, di)))

    for dop in DTYPES:
        for di in DTYPES:
            yield func, dop, di

def test_dtype2():
    @decorators.square
    class Op(Operator):

        def direct(self, input, output):
            np.multiply(input, input, output)
    op = Op()
    input = complex(1, 1)

    def func(di):
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = op(i)
        assert_eq(o.dtype, (i * i).dtype, str(di))
        assert_eq(o, i * i, str(di))
    for di in DTYPES:
        yield func, di


#===================
# Test operator name
#===================

def test_name():
    class sqrt(Operator):
        pass

    class MyOp(Operator):
        __name__ = 'sqrt'
    ops = [Operator(), Operator(direct=np.sqrt), MyOp(), Operator(name='sqrt'),
           sqrt()]
    names = ['Operator'] + 4 * ['sqrt']

    def func(op, name):
        assert op.__name__ == name
    for op, name in zip(ops, names):
        yield func, op, name


def test_merge_name():
    @decorators.linear
    class AbsorbOperator(Operator):

        def __init__(self, **keywords):
            Operator.__init__(self, **keywords)
            self.set_rule(('.', HomothetyOperator), lambda s, o: s.copy(),
                          CompositionOperator)

    class sqrt(AbsorbOperator):
        pass

    class MyOp(AbsorbOperator):
        __name__ = 'sqrt'
    ops = [AbsorbOperator(name='sqrt'), MyOp(), sqrt()]
    names = 3 * ['sqrt']

    def func(op, name):
        assert op.__name__ == name
    for (op, name), h in itertools.product(zip(ops, names),
                                           (I, HomothetyOperator(2))):
        yield func, op * h, name
        yield func, h * op, name


#=========================
# Test operator comparison
#=========================

def test_eq():
    def func(op1, op2):
        assert_eq(op1, op2)
    for cls in OPS:
        yield func, cls(), cls()


#================
# Test iadd, imul
#================

def test_iadd_imul():

    def func(op1, op2, operation):
        if operation is operator.iadd:
            op = op1 + op2
            op1 += op2
        else:
            op = op1 * op2.T
            op1 *= op2.T
        assert_eq(op1, op)
    for operation in (operator.iadd, operator.imul):
        for cls2 in OPS:
            for cls1 in OPS:
                yield func, cls1(), cls2(), operation


#===========================
# Test attribute propagation
#===========================

def test_propagation_attribute1():
    @decorators.square
    class AddAttribute(Operator):
        attrout = {'newattr_direct': True}
        attrin = {'newattr_transpose': True}

        def direct(self, input, output):
            pass

        def transpose(self, input, output):
            pass

    @decorators.square
    class AddAttribute2(Operator):
        attrout = {'newattr_direct': False}
        attrin = {'newattr_transpose': False}

        def direct(self, input, output):
            pass

        def transpose(self, input, output):
            pass

    @decorators.square
    class AddAttribute3(Operator):
        attrout = {'newattr3_direct': True}
        attrin = {'newattr3_transpose': True}

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
        assert_eq(op.attrout, {'newattr_direct': False})
        assert_eq(op.attrin, {'newattr_transpose': True})
        assert op.T(i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        assert op(i).newattr_direct
        assert op(i).newattr3_direct
        assert_eq(op.attrout, {'newattr_direct': True,
                               'newattr3_direct': True})
        assert_eq(op.attrin, {'newattr_transpose': True,
                              'newattr3_transpose': True})
        assert op.T(i).newattr_transpose
        assert op.T(i).newattr3_transpose
    for i in inputs:
        yield func1, i

    def func2(i_):
        print
        print 'op'
        print '=='
        op = AddAttribute()
        i = i_.copy()
        assert op(i, i).newattr_direct
        i = i_.copy()
        assert op.T(i, i).newattr_transpose

        pool.clear()
        print
        print 'op2 * op'
        print '======='
        op = AddAttribute2() * AddAttribute()
        i = i_.copy()
        assert not op(i, i).newattr_direct
        i = i_.copy()
        assert op.T(i, i).newattr_transpose

        pool.clear()
        print
        print 'op3 * op'
        print '======='
        op = AddAttribute3() * AddAttribute()
        i = i_.copy()
        o = op(i, i)
        assert o.newattr_direct
        assert o.newattr3_direct
        i = i_.copy()
        o = op.T(i, i)
        assert o.newattr_transpose
        assert o.newattr3_transpose
    for i_ in inputs:
        yield func2, i_


def test_propagation_attribute2():
    @decorators.square
    class Op(Operator):
        attrin = {'attr_class': 1, 'attr_instance': 2, 'attr_other': 3}
        attrout = {'attr_class': 4, 'attr_instance': 5, 'attr_other': 6}

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
    assert output.__dict__ == {'attr_instance': 10, 'attr_instance1': 11,
                               'attr_instance2': 12, 'attr_class': 30,
                               'attr_class2': 2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance2': 42,
                               'attr_class': 30, 'attr_class2': 32}

    op = Op().T
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance1': 41,
                               'attr_class': 30, 'attr_class1': 31}
    input = ndarray2(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {'attr_instance': 10, 'attr_instance2': 12,
                               'attr_instance1': 11, 'attr_class': 30,
                               'attr_class1': 1}

    op = Op().T * Op()  # -> ndarray2 -> ndarray1
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance1': 41,
                               'attr_class': 30, 'attr_class1': 1}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance1': 11,
                               'attr_instance2': 42, 'attr_class': 30,
                               'attr_class1': 1}

    op = Op() * Op().T  # -> ndarray1 -> ndarray2
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance2': 12,
                               'attr_instance1': 41, 'attr_class': 30,
                               'attr_class2': 2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance': 40, 'attr_instance2': 42,
                               'attr_class': 30, 'attr_class2': 2}


#=======================
# Test class propagation
#=======================

def check_propagation_class(op, i, c):
    o = op(i)
    assert_is(type(o), c)


def check_propagation_class_inplace(op, i, c):
    i = i.copy()
    op(i, i)
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
        o = op1 * op2
        assert_is(o(1).__class__, expected)

    def func3(op1, op2, op3, expected):
        o = op1 * op2 * op3
        assert_is(o(1).__class__, expected)

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

def test_comm_composite():
    comms_all = (None, MPI.COMM_SELF, MPI.COMM_WORLD)

    def func(cls, comms, inout):
        ops = [Operator(**{'comm'+inout: c}) for c in comms]
        keywords = {}
        args = ()
        if cls in (BlockDiagonalOperator, BlockRowOperator):
            keywords = {'axisin': 0}
        elif cls is BlockColumnOperator:
            keywords = {'axisout': 0}
        else:
            keywords = {}
        if MPI.COMM_SELF in comms and MPI.COMM_WORLD in comms:
            assert_raises(ValueError, cls, ops, *args, **keywords)
            return
        op = cls(ops, *args, **keywords)
        assert_is(getattr(op, 'comm'+inout), first_is_not(comms, None))
    for cls in (AdditionOperator, MultiplicationOperator, BlockRowOperator,
                BlockDiagonalOperator, BlockColumnOperator):
        for comms in itertools.combinations_with_replacement(comms_all, 3):
            for inout in ('in', 'out'):
                yield func, cls, comms, inout
if PYTHON_26:
    test_comm_composite = skiptest(test_comm_composite)


def test_comm_composition():
    comms_all = (None, MPI.COMM_SELF, MPI.COMM_WORLD)

    def func(commin, commout):
        ops = [Operator(commin=commin), Operator(commout=commout)]
        if None not in (commin, commout) and commin is not commout:
            assert_raises(ValueError, CompositionOperator, ops)
            return
        op = CompositionOperator(ops)
        assert_is(op.commin, commin)
        assert_is(op.commout, commout)
    for commin, commout in itertools.product(comms_all, repeat=2):
        yield func, commin, commout


def test_comm_propagation():
    composite = (AdditionOperator, MultiplicationOperator, BlockRowOperator,
                 BlockDiagonalOperator, BlockColumnOperator)
    commin = commin_ = MPI.COMM_WORLD.Dup()
    commout = commout_ = MPI.COMM_WORLD.Dup()

    class OpGetComm(Operator):
        def propagate_commin(self, comm):
            return OpNewComm(commin=comm, commout=comm)

    class OpNewComm(Operator):
        pass

    class OpSetComm1(Operator):
        commin = commin_
        commout = commout_

    class OpSetComm2(Operator):
        commin = commin_
        commout = commin_

    opgetcomm = OpGetComm()
    opsetcomm1 = OpSetComm1()
    opsetcomm2 = OpSetComm2()

    # get+set in composition
    def func1(i, op):
        assert_is(op.commin, commin)
        assert_is(op.commout, commout)
        opget = op.operands[i]
        assert_is_instance(opget, OpNewComm)
        if i == 0:
            assert_is(opget.commin, commout)
            assert_is(opget.commout, commout)
        else:
            assert_is(opget.commin, commin)
            assert_is(opget.commout, commin)

    for i, ops in enumerate([(opgetcomm, opsetcomm1),
                             (opsetcomm1, opgetcomm)]):
        op = CompositionOperator(ops)
        yield func1, i, op

    # get+set in composite
    def func2(i, op):
        assert_is(op.commin, commin)
        assert_is(op.commout, commin)
        opget = op.operands[i]
        assert_is_instance(opget, OpNewComm)
        assert_is(opget.commin, commin)
        assert_is(opget.commout, commin)

    for cls in composite:
        for i, ops in enumerate([(opgetcomm, opsetcomm2),
                                 (opsetcomm2, opgetcomm)]):
            keywords = {}
            if cls in (BlockDiagonalOperator, BlockRowOperator):
                keywords = {'axisin': 0}
            elif cls is BlockColumnOperator:
                keywords = {'axisout': 0}
            op = cls(ops, **keywords)
            yield func2, i, op

    # composition(get) + set in composite
    def func3(i, op):
        assert_is(op.commin, commin)
        assert_is(op.commout, commin)
        compget = op.operands[i]
        assert_is(compget.commin, commin)
        assert_is(compget.commout, commin)
        opget = op.operands[i].operands[i]
        assert_is_instance(opget, OpNewComm)
        assert_is(opget.commin, commin)
        assert_is(opget.commout, commin)
    for cls in composite:
        for i, ops in enumerate([(opgetcomm*Operator(), opsetcomm2),
                                 (opsetcomm2, Operator()*opgetcomm)]):
            keywords = {}
            if cls in (BlockDiagonalOperator, BlockRowOperator):
                keywords = {'axisin': 0}
            elif cls is BlockColumnOperator:
                keywords = {'axisout': 0}
            op = cls(ops, **keywords)
            yield func3, i, op

    # composite(set) + get in composition

    def func4(i, op):
        assert_is(op.commin, commin)
        assert_is(op.commout, commin)
        opget = op.operands[i]
        assert_is_instance(opget, OpNewComm)
        assert_is(opget.commin, commin)
        assert_is(opget.commout, commin)
    for cls in composite:
        keywords = {}
        if cls in (BlockDiagonalOperator, BlockRowOperator):
            keywords = {'axisin': 0}
        elif cls is BlockColumnOperator:
            keywords = {'axisout': 0}
        for ops_in in [(opsetcomm2, Operator()), (Operator(), opsetcomm2)]:
            op_in = cls(ops_in, **keywords)
            for i, op in enumerate([opgetcomm*op_in, op_in*opgetcomm]):
                yield func4, i, op

    # composite(get) + set in composition
    def func5(i, j, op):
        assert_is(op.commin, commin)
        assert_is(op.commout, commin)
        compget = op.operands[j]
        assert_is(compget.commin, commin)
        assert_is(compget.commout, commin)
        opget = compget.operands[i]
        assert_is_instance(opget, OpNewComm)
        assert_is(opget.commin, commin)
        assert_is(opget.commout, commin)
    for cls in composite:
        keywords = {}
        if cls in (BlockDiagonalOperator, BlockRowOperator):
            keywords = {'axisin': 0}
        elif cls is BlockColumnOperator:
            keywords = {'axisout': 0}
        for i, ops_in in enumerate([(opgetcomm, Operator()),
                                    (Operator(), opgetcomm)]):
            op_in = cls(ops_in, **keywords)
            for j, op in enumerate([op_in*opsetcomm2, opsetcomm2*op_in]):
                yield func5, i, j, op


#===========================
# Test in-place/out-of-place
#===========================

def test_inplace1():
    @decorators.square
    class NotInplace(Operator):

        def direct(self, input, output):
            output[...] = 0
            output[0] = input[0]
    pool.clear()
    op = NotInplace()
    v = np.array([2., 0., 1.])
    op(v, v)
    assert_eq(v, [2, 0, 0])
    assert_eq(len(pool), 1)


@skiptest
def test_inplace_can_use_output():
    A = zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0]: 'A',
           B.__array_interface__['data'][0]: 'B',
           C.__array_interface__['data'][0]: 'C',
           D.__array_interface__['data'][0]: 'D'}

    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace': inplace})
            self.inplace = inplace
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

    def show_pool():
        result = ''
        for s in pool:
            try:
                result += ids[s.__array_interface__['data'][0]]
            except:
                result += 'U'
        return result

    expecteds_outplace = {
        2: ['BBA',     # II
            'BBA',     # IO
            'BCA',     # OI
            'BCA'],    # OO
        3: ['BBBA',    # III
            'BBBA',    # IIO
            'BBCA',    # IOI
            'BBCA',    # IOO
            'BCCA',    # OII
            'BCCA',    # OIO
            'BCBA',    # OOI
            'BCBA'],   # OOO
        4: ['BBBBA',   # IIII
            'BBBBA',   # IIIO
            'BBBCA',   # IIOI
            'BBBCA',   # IIOO
            'BBCCA',   # IOII
            'BBCCA',   # IOIO
            'BBCBA',   # IOOI
            'BBCBA',   # IOOO
            'BCCCA',   # OIII
            'BCCCA',   # OIIO
            'BCCBA',   # OIOI
            'BCCBA',   # OIOO
            'BCBBA',   # OOII
            'BCBBA',   # OOIO
            'BCBCA',   # OOOI
            'BCBCA']}  # OOOO

    expecteds_inplace = {
        2: ['AAA',     # II
            'ABA',     # IO
            'ABA',     # OI
            'ABA'],    # OO
        3: ['AAAA',    # III
            'ABBA',    # IIO
            'ABAA',    # IOI
            'AABA',    # IOO
            'ABAA',    # OII
            'ABBA',    # OIO
            'ABAA',    # OOI
            'ACBA'],   # OOO
        4: ['AAAAA',   # IIII
            'ABBBA',   # IIIO
            'ABBAA',   # IIOI
            'AAABA',   # IIOO
            'ABAAA',   # IOII
            'AABBA',   # IOIO
            'AABAA',   # IOOI
            'ABABA',   # IOOO
            'ABAAA',   # OIII
            'ABBBA',   # OIIO
            'ABBAA',   # OIOI
            'ABABA',   # OIOO
            'ABAAA',   # OOII
            'ABABA',   # OOIO
            'ABABA',   # OOOI
            'ABABA']}  # OOOO

    def func_outplace(n, i, expected, strops):
        pool._buffers = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_pool = show_pool  # debug
        v = A[:8].view(float)
        v[0] = 1
        w = B[:(n+1)*8].view(float)
        op(v, w)
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_pool(), 'CD', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    def func_inplace(n, i, expected, strops):
        pool._buffers = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        v = A[:8].view(float)
        v[0] = 1
        w = A[:(n+1)*8].view(float)
        op(v, w)
        log = ''.join(log) + 'A'
        assert_eq(log, expected, strops)
        assert_eq(show_pool(), 'BC', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_eq(w, w2, strops)

    # prevent memory manager from allocating buffer: only use the provided pool
    memory.MEMORY_TOLERANCE = np.inf
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


@skiptest
def test_inplace_cannot_use_output():
    A = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0]: 'A',
           B.__array_interface__['data'][0]: 'B',
           C.__array_interface__['data'][0]: 'C',
           D.__array_interface__['data'][0]: 'D'}

    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace': inplace})
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
        return ''.join([ids[s.__array_interface__['data'][0]] for s in pool])

    expecteds_outplace = {
        2: ['BCA',     # II
            'BCA',     # IO
            'BCA',     # OI
            'BCA'],    # OO
        3: ['BCCA',    # III
            'BCCA',    # IIO
            'BDCA',    # IOI
            'BDCA',    # IOO
            'BCCA',    # OII
            'BCCA',    # OIO
            'BDCA',    # OOI
            'BDCA'],   # OOO
        4: ['BCCCA',   # IIII
            'BCCCA',   # IIIO
            'BDDCA',   # IIOI
            'BDDCA',   # IIOO
            'BDCCA',   # IOII
            'BDCCA',   # IOIO
            'BCDCA',   # IOOI
            'BCDCA',   # IOOO
            'BCCCA',   # OIII
            'BCCCA',   # OIIO
            'BDDCA',   # OIOI
            'BDDCA',   # OIOO
            'BDCCA',   # OOII
            'BDCCA',   # OOIO
            'BCDCA',   # OOOI
            'BCDCA']}  # OOOO

    expecteds_inplace = {
        2: ['ABA',     # II
            'ABA',     # IO
            'ABA',     # OI
            'ABA'],    # OO
        3: ['ABBA',    # III
            'ABBA',    # IIO
            'ACBA',    # IOI
            'ACBA',    # IOO
            'ABBA',    # OII
            'ABBA',    # OIO
            'ACBA',    # OOI
            'ACBA'],   # OOO
        4: ['ABBBA',   # IIII
            'ABBBA',   # IIIO
            'ACCBA',   # IIOI
            'ACCBA',   # IIOO
            'ACBBA',   # IOII
            'ACBBA',   # IOIO
            'ABCBA',   # IOOI
            'ABCBA',   # IOOO
            'ABBBA',   # OIII
            'ABBBA',   # OIIO
            'ACCBA',   # OIOI
            'ACCBA',   # OIOO
            'ACBBA',   # OOII
            'ACBBA',   # OOIO
            'ABCBA',   # OOOI
            'ABCBA']}  # OOOO

    def func_outplace(n, i, expected, strops):
        pool._buffers = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float)
        v[:] = range(n+1)
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
        pool._buffers = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float)
        v[:] = range(n+1)
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


#====================
# Test associativity
#====================

def test_associativity():

    class Op1(Operator):
        pass

    class Op2(Operator):
        pass

    class Op3(Operator):
        pass

    class Op4(Operator):
        pass

    # composite and operator
    def func1(cls, op):
        assert_is_instance(op, cls)
        assert_eq(len(op.operands), 3)
        if all(isinstance(o, c) for o, c in zip(op.operands, [Op2, Op3, Op1])):
            raise SkipTest()  # commutative rules do not preserve order...
        for o, c in zip(op.operands, [Op1, Op2, Op3]):
            assert_is_instance(o, c)
    for operation in (AdditionOperator, MultiplicationOperator,
                      CompositionOperator):
        yield func1, operation, operation([operation([Op1(), Op2()]), Op3()])
        yield func1, operation, operation([Op1(), operation([Op2(), Op3()])])

    # composite and composite
    def func2(cls, op):
        assert_is_instance(op, cls)
        assert_eq(len(op.operands), 4)
        for o, c in zip(op.operands, [Op1, Op2, Op3, Op4]):
            assert_is_instance(o, c)
    for operation in (AdditionOperator, MultiplicationOperator,
                      CompositionOperator):
        yield func2, operation, operation([operation([Op1(), Op2()]),
                                           operation([Op3(), Op4()])])

    a = GroupOperator([Op1(), Op2()])
    b = GroupOperator([Op3(), Op4()])

    def func3(o1, o2):
        op = o1 * o2
        assert_is_instance(op, CompositionOperator)
        assert_eq(len(op.operands), 2)
        assert_is(op.operands[0], o1)
        assert_is(op.operands[1], o2)
    for o1, o2 in [(Op1(), a), (a, Op1()), (a, b)]:
        yield func3, o1, o2


#==================
# Test commutative
#==================

def test_addition():
    @decorators.square
    class Op(Operator):

        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)

        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.sum([Op(v) for v in [1]])
    assert_is(op.__class__, Op)

    op = np.sum([Op(v) for v in [1, 2]])
    assert_eq(op.__class__, AdditionOperator)

    pool.clear()
    assert_eq(op(1), 3)
    assert_eq(len(pool), 1)

    op = np.sum([Op(v) for v in [1, 2, 4]])
    assert_is(op.__class__, AdditionOperator)

    pool.clear()
    input = np.array(1, int)
    output = np.array(0, int)
    assert_eq(op(input, output), 7)
    assert_eq(input, 1)
    assert_eq(output, 7)
    assert_eq(len(pool), 1)

    pool.clear()
    output = input
    assert_eq(op(input, output), 7)
    assert_eq(input, 7)
    assert_eq(output, 7)
    assert_eq(len(pool), 2)


def test_addition_flags():
    def func(f):
        o = AdditionOperator([Operator(flags=f), Operator(flags=f)])
        assert getattr(o.flags, f)
    for f in 'linear,real,square,symmetric,hermitian,separable'.split(','):
        yield func, f


def test_multiplication():
    @decorators.square
    class Op(Operator):

        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)

        def direct(self, input, output):
            np.multiply(input, self.v, output)

    pool.clear()
    op = MultiplicationOperator([Op(v) for v in [1]])
    assert_is(op.__class__, Op)

    op = MultiplicationOperator([Op(v) for v in [1,2]])
    assert_eq(op.__class__, MultiplicationOperator)
    assert_eq(op(1), 2)
    assert_eq(len(pool), 1)

    op = MultiplicationOperator([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, MultiplicationOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(pool), 1)

    output = input
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(output, 8)
    assert_eq(len(pool), 2)


def test_multiplication_flags():

    def func(f):
        o = MultiplicationOperator([Operator(flags=f), Operator(flags=f)])
        assert getattr(o.flags, f)
    for f in 'real,square,separable'.split(','):
        yield func, f


def test_commutative_shapes():

    def func(cls, OP1, OP2):
        n1 = OP1.__name__
        n2 = OP2.__name__
        op = cls([OP1(), OP2()])

        shape_output = op.flags.shape_output
        if 'Expl' in (n1[:4], n2[:4]):
            assert shape_output == 'explicit'
        elif n1[4:] == 'Expl' and n2[:4] == 'Impl' or \
             n2[4:] == 'Expl' and n1[:4] == 'Impl':
            assert shape_output == 'explicit'
        elif 'Impl' in (n1[:4], n2[:4]):
            assert shape_output == 'implicit'
        else:
            assert shape_output == 'unconstrained'

        shape_input = op.flags.shape_input
        if 'Expl' in (n1[4:], n2[4:]):
            assert shape_input == 'explicit'
        elif n1[:4] == 'Expl' and n2[4:] == 'Impl' or \
             n2[:4] == 'Expl' and n1[4:] == 'Impl':
            assert shape_input == 'explicit'
        elif 'Impl' in (n1[4:], n2[4:]):
            assert shape_input == 'implicit'
        else:
            assert shape_input == 'unconstrained'

    for cls in (AdditionOperator, MultiplicationOperator):
        for OP1, OP2 in itertools.product(OPS, repeat=2):
            yield func, cls, OP1, OP2


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
    for ndim in range(1, 5):
        for nops in range(1, 5):
            for Op in [HomothetyOperator, HomothetyOutplaceOperator]:
                slices_ = [
                    [distribute_slice(size, i, nops) for i in range(nops)],
                    [distribute_slice(size, i, size) for i in range(nops)],
                    [ndim * [slice(i, None, nops)] for i in range(nops)]]
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


def test_block_slice_rule_homothety():
    b = BlockSliceOperator(2*[HomothetyOperator(3)],
                           [slice(0, 10), slice(12, 14)])
    hb = HomothetyOperator(2) * b
    assert_is_instance(hb, BlockSliceOperator)
    for op in hb.operands:
        assert_is_instance(op, HomothetyOperator)
        assert_eq(op.data, 6)


#==================
# Test composition
#==================

def test_composition1():

    def func(op, shapein, shapeout):
        assert_eq(op.shapein, shapein)
        assert_eq(op.shapeout, shapeout)
        if shapein is not None and shapein == shapeout:
            assert_flags(op, 'square')
    for shapein in SHAPES:
        for shapemid in SHAPES:
            if shapemid is None and shapein is not None:
                continue
            op1 = Operator(shapein=shapein, shapeout=shapemid)
            for shapeout in SHAPES:
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
    for shape in SHAPES:
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

    pool.clear()
    op = np.product([Op(v) for v in [1]])
    assert_is(op.__class__, Op)
    op(1)
    assert_eq(len(pool), 0)

    pool.clear()
    op = np.product([Op(v) for v in [1, 2]])
    assert_is(op.__class__, CompositionOperator)
    assert_eq(op(1), 2)
    assert_eq(len(pool), 0)

    pool.clear()
    assert_eq(op([1]), 2)
    assert_eq(len(pool), 0)

    op = np.product([Op(v) for v in [1, 2, 4]])
    assert_is(op.__class__, CompositionOperator)

    pool.clear()
    input = np.array(1, int)
    output = np.array(0, int)
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(pool), 0)

    pool.clear()
    output = input
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(len(pool), 0)

    pool.clear()
    input = np.array([1], int)
    output = np.array([0], int)
    assert_eq(op(input, output), 8)
    assert_eq(input, 1)
    assert_eq(output, 8)
    assert_eq(len(pool), 0)

    pool.clear()
    output = input
    assert_eq(op(input, output), 8)
    assert_eq(input, 8)
    assert_eq(output, 8)
    assert_eq(len(pool), 0)


def test_composition_flags():
    def func1(f):
        o = CompositionOperator([Operator(flags=f), Operator(flags=f)])
        assert getattr(o.flags, f)
    for f in 'linear,real,square,separable'.split(','):
        yield func1, f

    def func2(f):
        o = CompositionOperator([Operator(), Operator(flags=f)])
        assert getattr(o.flags, f)
    for f in 'aligned_input,contiguous_input'.split(','):
        yield func2, f

    def func3(f):
        o = CompositionOperator([Operator(flags=f), Operator()])
        assert getattr(o.flags, f)
    for f in 'aligned_output,contiguous_output'.split(','):
        yield func3, f

    def func4(f):
        o = CompositionOperator([Operator(), Operator()])
        assert not getattr(o.flags, f)
        o = CompositionOperator([OperatorIR(), Operator()])
        assert getattr(o.flags, f)
    yield func4, 'inplace_reduction'


def test_composition_shapes():
    def func(OP1, OP2):
        n1 = OP1.__name__
        n2 = OP2.__name__
        if n1[4:] == 'Expl' and n2[:4] == 'Expl':
            op = OP1() * OP2(shapeout=3)
        else:
            op = OP1() * OP2()

        shape_output = op.flags.shape_output
        if n1[:4] == 'Unco':
            assert shape_output == 'unconstrained'
        elif n1[:4] == 'Expl':
            assert shape_output == 'explicit'
        elif n2[:4] == 'Expl':
            assert shape_output == 'explicit'
        elif n2[:4] == 'Impl':
            assert shape_output == 'implicit'
        else:
            assert shape_output == 'unconstrained'

        shape_input = op.flags.shape_input
        if n2[4:] == 'Unco':
            assert shape_input == 'unconstrained'
        elif n2[4:] == 'Expl':
            assert shape_input == 'explicit'
        elif n1[4:] == 'Expl':
            assert shape_input == 'explicit'
        elif n1[4:] == 'Impl':
            assert shape_input == 'implicit'
        else:
            assert shape_input == 'unconstrained'

    for OP1, OP2 in itertools.product(OPS, repeat=2):
        yield func, OP1, OP2


def test_composition_get_requirements():
    @decorators.inplace
    class I__(Operator):
        pass

    @decorators.aligned
    @decorators.contiguous
    class IAC(I__):
        pass

    class O____(Operator):
        pass

    @decorators.aligned_input
    @decorators.contiguous_input
    class O__AC(O____):
        pass

    @decorators.aligned_output
    @decorators.contiguous_output
    class OAC__(O____):
        pass

    @decorators.aligned
    @decorators.contiguous
    class OACAC(O____):
        pass

    Is = [I__(), IAC()]
    Os = [O____(), O__AC(), OAC__(), OACAC()]

    tests ={'I'  : [[0]],
            'O'  : [[0], []],
            'II' : [[0, 1]],
            'IO' : [[0, 1], []],
            'OI' : [[0], [1]],
            'OO' : [[0], [1], []],
            'III': [[0, 1, 2]],
            'IIO': [[0, 1, 2], []],
            'IOI': [[0, 1], [2]],
            'IOO': [[0, 1], [2], []],
            'OII': [[0], [1, 2]],
            'OIO': [[0], [1, 2], []],
            'OOI': [[0], [1], [2]],
            'OOO': [[0], [1], [2], []]}

    def get_requirements(ops, t, g):
        rn = [len(_) for _ in g]
        for i in range(len(rn)-1):
            rn[i] -= 1

        ra = [max(ops[i].flags.aligned_output for i in g[0])] + \
             [max([ops[_[0]-1].flags.aligned_input] +
                  [ops[i].flags.aligned_output for i in _]) for _ in g[1:-1]]+\
             ([max(ops[i].flags.aligned_input for i in range(t.rfind('O'),
              len(ops)))] if len(g) > 1 else [])
        rc = [max(ops[i].flags.contiguous_output for i in g[0])] + \
             [max([ops[_[0]-1].flags.contiguous_input] +
                 [ops[i].flags.contiguous_output for i in _])for _ in g[1:-1]]+\
             ([max(ops[i].flags.contiguous_input for i in  range(t.rfind('O'),
              len(ops)))] if len(g) > 1 else [])
        return rn, ra, rc

    c = CompositionOperator(Is)

    def func(t, rn1, rn2, ra1, ra2, rc1, rc2):
        assert rn1 == rn2
        assert ra1 == ra2
    for t, g in tests.items():
        it = [Is if _ == 'I' else Os for _ in t]
        for ops in itertools.product(*it):
            c.operands = ops
            rn1, ra1, rc1 = c._get_requirements()
            rn2, ra2, rc2 = get_requirements(ops, t, g)
            yield func, t, rn1, rn2, ra1, ra2, rc1, rc2


#=============
# Test powers
#=============

def test_powers():

    def func(op, n):
        p = op ** n
        if n < -1:
            assert_is_instance(p, CompositionOperator)
            for o in p.operands:
                assert_is(o, op.I)
        elif n == -1:
            assert_is(p, op.I)
        elif n == 0:
            assert_is_instance(p, IdentityOperator)
        elif n == 1:
            assert_is(p, op)
        else:
            assert_is_instance(p, CompositionOperator)
            for o in p.operands:
                assert_is(o, op)
    for op in [SquareOp(), SquareOp(shapein=3)]:
        for n in range(-3, 4):
            yield func, op, n


def test_powers_diagonal():
    diag = np.array([1., 2, 3])
    d = DiagonalOperator(diag)

    def func(n):
        assert_eq((d**n).todense(), DiagonalOperator(diag**n).todense())
    for n in (-1.2, -1, -0.5, 0, 0.5, 1, 2.4):
        yield func, n


#=============================
# Test broadcasting operators
#=============================

def test_diagonal1():
    data = (0., 1., [0, 0], [1, 1], 2, [2, 2], [0, 1], [-1, -1], [-1, 1],
            [2, 1])
    expected = (ZeroOperator, IdentityOperator, ZeroOperator, IdentityOperator,
                HomothetyOperator, HomothetyOperator, MaskOperator,
                HomothetyOperator, DiagonalOperator, DiagonalOperator)

    def func(d, e):
        op = DiagonalOperator(d)
        if all(_ in (-1, 1) for _ in op.data.flat):
            assert op.flags.involutary
        assert_is(op.__class__, e)
    for d, e in zip(data, expected):
        yield func, d, e


def test_diagonal2():
    ops = (DiagonalOperator([1., 2], broadcast='rightward'),
           DiagonalOperator([[2., 3, 4], [5, 6, 7]], broadcast='rightward'),
           DiagonalOperator([1., 2, 3, 4, 5], broadcast='leftward'),
           DiagonalOperator(np.arange(20).reshape(4, 5), broadcast='leftward'),
           DiagonalOperator(np.arange(120.).reshape(2, 3, 4, 5)),
           HomothetyOperator(7.),
           IdentityOperator())

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
            for j in range(i, 7):
                d2 = ops[j]
                if set((d1.broadcast, d2.broadcast)) == \
                   set(('leftward', 'rightward')):
                    continue
                yield func, op, d1, d2


def test_masking():
    mask = MaskOperator(0)
    assert isinstance(mask, IdentityOperator)
    mask = MaskOperator(0, shapein=(32, 32), dtype=np.float32)
    assert isinstance(mask, IdentityOperator)
    assert mask.shapein == (32, 32)
    assert mask.dtype == np.float32

    mask = MaskOperator(1)
    assert isinstance(mask, ZeroOperator)
    mask = MaskOperator(1, shapein=(32, 32), dtype=np.float32)
    assert isinstance(mask, ZeroOperator)
    assert mask.shapein == (32, 32)
    assert mask.dtype == np.float32

    b = np.array([3., 4., 1., 0., 3., 2.])
    c = np.array([3., 4., 0., 0., 3., 0.])
    mask = MaskOperator(np.array([0, 0., 1., 1., 0., 1], dtype=np.int8))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([1, 1., 0., 0., 1., 0]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([False, False, True, True, False, True]))
    assert np.all(mask(b) == c)

    b = np.array([[3., 4.], [1., 0.], [3., 2.]])
    c = np.array([[3., 4.], [0., 0.], [3., 0.]])
    mask = MaskOperator(np.array([[0, 0.], [1., 1.], [0., 1.]], dtype='int8'))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([[1, 1.], [0., 0.], [1., 0.]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([[False, False],
                                  [True, True],
                                  [False, True]]))
    assert np.all(mask(b) == c)

    b = np.array([[[3, 4.], [1., 0.]], [[3., 2], [-1, 9]]])
    c = np.array([[[3, 4.], [0., 0.]], [[3., 0], [0, 0]]])
    mask = MaskOperator(np.array([[[0, 0.], [1., 1.]],
                                  [[0., 1], [1, 1]]], int))
    assert np.all(mask(b) == c)

    mask = DiagonalOperator(np.array([[[1, 1], [0., 0]], [[1, 0], [0, 0]]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([[[False, False], [True, True]],
                                  [[False, True], [True, True]]]))
    assert np.all(mask(b) == c)

    c = mask(b, b)
    assert id(b) == id(c)


def test_masking2():
    m = MaskOperator([True, False, True])
    assert_eq(m * m,  m)


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

    s = HomothetyOperator(complex(1, 1))
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
        assert_eq(o, result, str((model, i)))
        assert_eq(o.dtype, int, str((model, i)))
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
    c = 1, np.array([1, 2]), np.array([2, 3, 4])
    t = 'scalar', 'rightward', 'leftward'

    def func(c1, t1, c2, t2):
        op2 = ConstantOperator(c2, broadcast=t2)
        op = op1 + op2
        if set((op1.broadcast, op2.broadcast)) != \
           set(('rightward', 'leftward')):
            assert_is_instance(op, ConstantOperator)
        v = np.zeros((2, 3))
        op(np.nan, v)
        z = np.zeros((2, 3))
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
    cs = (ConstantOperator(3),
          ConstantOperator([1, 2, 3], broadcast='leftward'),
          ConstantOperator(np.ones((2, 3))))
    os = (I, H(2, shapein=(2, 3)) * Operator(direct=np.square, shapein=(2, 3),
                                             flags='square'), H(5))
    results = (((H, 3), (C, (H, 6)), (H, 15)),
               ((D, [1, 2, 3]), (C, (D, [2, 4, 6])), (D, [5, 10, 15])),
               ((IdentityOperator, 1), (C, (H, 2)), (H, 5)))
    v = np.arange(6).reshape((2, 3))

    def func(c, o, r):
        op = MultiplicationOperator([c, o])
        assert_eq(op(v), c.data*o(v))
        assert_is(type(op), r[0])
        if type(op) is CompositionOperator:
            op = op.operands[0]
            r = r[1]
            assert_is(type(op), r[0])
        assert_eq, op.data, r[1]
    for c, rs in zip(cs, results):
        for o, r in zip(os, rs):
            yield func, c, o, r


def _test_constant_reduction3():
    @decorators.square
    class Op(Operator):

        def direct(self, input, output):
            output[...] = input + np.arange(input.size).reshape(input.shape)
    os = (Op(shapein=()), Op(shapein=4), Op(shapein=(2, 3, 4)))
    cs = (ConstantOperator(2), ConstantOperator([2], broadcast='leftward'),
          ConstantOperator(2*np.arange(8).reshape((2, 1, 4)),
                           broadcast='leftward'))
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
    assert_eq(z*z, z)


def test_zero8():
    class Op(Operator):
        pass
    o = Op()
    assert type(o + O) is Op


#========================
# Test ReductionOperator
#========================

def test_reduction_operator1():
    def func(f, s, a):
        op = ReductionOperator(f, axis=a)
        v = np.arange(product(s)).reshape(s)
        if isinstance(f, np.ufunc):
            if np.__version__ < '1.7' and a is None:
                expected = f.reduce(v.flat, 0)
            else:
                expected = f.reduce(v, a)
        else:
            expected = f(v, axis=a)
        assert_eq(op(v), expected)
        out = np.empty_like(expected)
        op(v, out)
        assert_eq(out, expected)
    for f in (np.add, np.multiply, np.min, np.max, np.sum, np.prod):
        for s in SHAPES[2:]:
            for a in [None] + list(range(len(s))):
                yield func, f, s, a


def test_reduction_operator2():
    for f in (np.cos, np.modf):
        assert_raises(TypeError, ReductionOperator, f)
    f = np.add

    def func(n, op):
        v = np.empty(n * [2])
        assert_raises(TypeError if n == 0 else ValueError, op, v)
    for a in (1, 2, 3):
        op = ReductionOperator(f, axis=a)
        for n in range(0, a+1):
            yield func, n, op


#=================
# Test asoperator
#=================

def test_asoperator_scalar():
    scalars = [np.array(1, d) for d in DTYPES]

    def func1(s):
        o = asoperator(s)
        assert_is_instance(o, HomothetyOperator)

    def func2(s):
        o = asoperator(s, constant=True)
        assert_is_instance(o, ConstantOperator)
    for s in scalars:
        yield func1, s
        yield func2, s


def test_asoperator_ndarray():
    values = ([[1]], [[1, 2], [2, 3]], np.matrix([[3, 2.], [0, 1]]))

    def func1(v):
        o = asoperator(v)
        assert_is_instance(o, DenseOperator)
        assert_eq(np.array(v).shape, o.shape)

    def func2(v):
        o = asoperator(v, constant=True)
        if isinstance(v, np.matrix):
            assert_is_instance(o, DenseOperator)
            assert_eq(np.array(v).shape, o.shape)
        else:
            assert_is_instance(o, ConstantOperator)
            assert_eq(np.array(v).shape, o.shapeout)
    for v in values:
        yield func1, v
        yield func2, v


def test_asoperator_func():
    f = lambda x: x**2
    o = asoperator(f)
    assert_is_instance(o, Operator)
    assert_flags(o, 'inplace')

    def func(v):
        assert_eq(o(v), f(np.array(v)))
    for v in (2, [2], [2, 3]):
        yield func, v


def test_asoperator_ufunc():
    assert_raises(TypeError, asoperator, np.maximum)
    o = asoperator(np.cos)
    assert_flags(o, 'real,inplace,square,separable')
