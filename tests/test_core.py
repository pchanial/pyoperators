import operator

import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy.sparse import csc_matrix

from pyoperators import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    BlockSliceOperator,
    CompositionOperator,
    ConstantOperator,
    DenseBlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    GroupOperator,
    HomothetyOperator,
    I,
    IdentityOperator,
    MultiplicationOperator,
    Operator,
    PowerOperator,
    ReciprocalOperator,
    ReductionOperator,
    SparseOperator,
    SquareOperator,
    X,
    asoperator,
    flags,
)
from pyoperators.core import CopyOperator
from pyoperators.core import _pool as pool
from pyoperators.rules import rule_manager
from pyoperators.utils import isscalarlike, operation_assignment, product, split

from .common import ALL_OPS, DTYPES, OPS, CanUpdateOutput, HomothetyOutplace, totuple

np.seterr(all='raise')


def assert_is_inttuple(shape, msg=''):
    msg = f'{shape} is not an int tuple.' + (' ' + msg if msg else '')
    assert type(shape) is tuple, msg
    assert all([isinstance(s, int) for s in shape]), msg


def assert_square(op):
    assert op.flags.square
    assert op.shapein == op.shapeout


SHAPES = (None, (), (1,), (3,), (2, 3))


class OperatorNIR1(Operator):
    def direct(self, input, output):
        output[...] = input


class OperatorNIR2(Operator):
    def direct(self, input, output, operation=operation_assignment):
        operation(output, input)


# ===========
# Test flags
# ===========


@pytest.mark.parametrize('op', ALL_OPS)
def test_flags(op):
    try:
        o = op()
    except:
        try:
            v = np.arange(10.0)
            o = op(v)
        except:
            print('Cannot test: ' + op.__name__)
            return
    if type(o) is not op:
        print('Cannot test: ' + op.__name__)
        return
    if o.flags.idempotent:
        assert o is o(o)
    if o.flags.real:
        assert o is o.C
    if o.flags.symmetric:
        assert o is o.T
    if o.flags.hermitian:
        assert o is o.H
    if o.flags.involutary:
        assert o is o.I
    if o.flags.orthogonal:
        assert o.T is o.I
    if o.flags.unitary:
        assert o.H is o.I


def test_symmetric():
    mat = np.matrix([[2, 1], [1, 2]])

    @flags.square
    @flags.symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=2, dtype=mat.dtype)

        def direct(self, input, output):
            output[...] = np.dot(mat, input)

    op = Op()
    assert op.flags.linear
    assert op.flags.square
    assert op.flags.real
    assert op.flags.symmetric
    assert_equal(op.shape, (2, 2))
    assert_equal(op.shapeout, (2,))
    assert op is op.C
    assert op is op.T
    assert op is op.H
    assert_equal(op([1, 1]), np.array(mat * [[1], [1]]).ravel())


@pytest.mark.parametrize('cls', OPS)
def test_shape_input_and_output(cls):
    op = cls()
    flags = op.flags
    name = type(op).__name__
    kind = {'Expl': 'explicit', 'Impl': 'implicit', 'Unco': 'unconstrained'}
    assert_equal(flags.shape_output, kind[name[:4]])
    assert_equal(flags.shape_input, kind[name[4:]])


@pytest.mark.parametrize('cls', [OperatorNIR1, OperatorNIR2])
def test_update_output1(cls):
    op = cls()
    assert not op.flags.update_output
    out = np.zeros(3, dtype=int)
    with pytest.raises(ValueError):
        op([1, 0, 0], out, operation=operator.iadd)


def test_update_output2():
    assert CanUpdateOutput().flags.update_output
    with pytest.raises(ValueError):
        CanUpdateOutput()([1, 0, 0], operation=operator.iadd)


@pytest.mark.parametrize(
    'operation, expected',
    [
        (operation_assignment, [0, 1, 1]),
        (operator.iadd, [2, 5, 3]),
        (operator.imul, [0, 2, 0]),
    ],
)
def test_update_output3(operation, expected):
    op = CanUpdateOutput()
    inputs = [1, 1, 0], [0, 2, 1], [0, 1, 1]
    output = np.ones(3, dtype=int)
    for input in inputs:
        op(input, output, operation=operation)
    assert_equal(output, expected)


@pytest.mark.parametrize('flag', ['shape_input', 'shape_output'])
def test_autoflags(flag):
    with pytest.raises(ValueError):
        Operator(flags=flag)


# =============
# Test direct
# =============


@pytest.mark.parametrize(
    'ufunc, dtype',
    [
        (np.cos, np.float64),
        (np.invert, None),
        (np.negative, None),
    ],
)
def test_ufuncs(ufunc, dtype):
    op = Operator(ufunc)
    assert op.flags.real
    assert op.flags.inplace
    assert op.flags.outplace
    assert op.flags.square
    assert op.flags.separable
    assert op.dtype == dtype


def test_ufuncs_error():
    with pytest.raises(TypeError):
        Operator(np.maximum)


# ==================
# Test conjugation
# ==================


def test_conjugation():
    @flags.square
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

    for opT, opH, opIT, opIH in [
        (Op1T(), Op1H(), Op1IT(), Op1IH()),
        (Op2T(), Op2H(), Op2IT(), Op2IH()),
    ]:
        assert_equal(opT.C.todense(), dense.conj())
        assert_equal(opT.T.todense(), dense.T)
        assert_equal(opT.H.todense(), dense.T.conj())
        assert_equal(opH.C.todense(), dense.conj())
        assert_equal(opH.T.todense(), dense.T)
        assert_equal(opH.H.todense(), dense.T.conj())
        assert_equal(opIT.I.C.todense(), denseI.conj())
        assert_equal(opIT.I.T.todense(), denseI.T)
        assert_equal(opIT.I.H.todense(), denseI.T.conj())
        assert_equal(opIH.I.C.todense(), denseI.conj())
        assert_equal(opIH.I.T.todense(), denseI.T)
        assert_equal(opIH.I.H.todense(), denseI.T.conj())


# ==================
# Test *, / and **
# ==================

MATRIX_TIMES_MUL_OR_COMP = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
OPS_TIMES_MUL_OR_COMP = [
    2,
    [1, 2, 3],
    np.array(3),
    np.ones(3),
    np.negative,
    np.sqrt,
    np.matrix(MATRIX_TIMES_MUL_OR_COMP),
    csc_matrix(MATRIX_TIMES_MUL_OR_COMP),
    DenseOperator(MATRIX_TIMES_MUL_OR_COMP),
    HomothetyOperator(3),
    SquareOperator(),
    X,
    X.T,
]


@pytest.mark.parametrize('x', OPS_TIMES_MUL_OR_COMP)
@pytest.mark.parametrize('y', OPS_TIMES_MUL_OR_COMP)
def test_times_mul_or_comp(x, y):
    def islinear(_):
        if isinstance(_, (np.matrix, csc_matrix)):
            return True
        if _ is np.sqrt:
            return False
        if _ is np.negative:
            return True
        if isscalarlike(_):
            return True
        return _.flags.linear

    if not isinstance(x, Operator) and not isinstance(y, Operator):
        return

    if isinstance(x, np.ndarray):
        if isinstance(x, np.matrix):
            x = DenseOperator(x)
        elif x.ndim > 0:
            x = DiagonalOperator(x)
    if isinstance(x, csc_matrix):
        x = SparseOperator(x)
    if (
        x is X.T
        and (y is np.sqrt or isinstance(y, SquareOperator))
        or y is X.T
        and not isscalarlike(x)
        and not isinstance(x, HomothetyOperator)
    ):
        with pytest.raises(TypeError):
            eval('x * y', {'x': x, 'y': y})
        return

    with rule_manager(none=True):
        z = x * y

    if x is X and y is X:
        assert type(z) is MultiplicationOperator
    elif x is X.T and y is X or x is X and y is X.T:
        assert type(z) is CompositionOperator
    elif x is X:
        if (
            np.isscalar(y)
            or isinstance(y, (list, np.ndarray, HomothetyOperator))
            and not isinstance(y, np.matrix)
        ):
            assert type(z) is CompositionOperator
        else:
            assert type(z) is MultiplicationOperator
    elif type(x) is list or type(x) is np.ndarray and x.ndim > 0:
        if y is X:
            assert type(z) is CompositionOperator
        elif islinear(y):
            assert_equal(z, asoperator(y).T(x))
        else:
            assert type(z) is MultiplicationOperator
    elif type(y) is list or type(y) is np.ndarray and y.ndim > 0:
        if x is X.T:
            assert type(z) is CompositionOperator
        elif islinear(x):
            assert_equal(z, asoperator(x)(y))
        else:
            assert type(z) is MultiplicationOperator
    elif islinear(x) and islinear(y):
        assert type(z) is CompositionOperator
    else:
        assert type(z) is MultiplicationOperator


@pytest.mark.parametrize('flag', [False, True])
def test_div(flag):
    op = 1 / Operator(flags={'linear': flag})
    assert type(op) is CompositionOperator
    assert type(op.operands[0]) is ReciprocalOperator
    assert type(op.operands[1]) is Operator


@pytest.mark.xfail(reason='reason: Unknown')
def test_div_fail():
    assert type(1 / SquareOperator()) is PowerOperator


def test_pow():
    data = [[1, 1], [0, 1]]
    op_lin = DenseOperator(data)
    assert_equal((op_lin**3).data, np.dot(np.dot(data, data), data))
    op_nl = ConstantOperator(data)
    assert_equal((op_nl**3).data, data)


@flags.linear
@flags.square
class SquareOp(Operator):
    pass


@pytest.mark.parametrize('op', [SquareOp(), SquareOp(shapein=3)])
@pytest.mark.parametrize('n', range(-3, 4))
def test_pow2(op, n):
    p = op**n
    if n < -1:
        assert isinstance(p, CompositionOperator)
        for o in p.operands:
            assert o is op.I
    elif n == -1:
        assert p is op.I
    elif n == 0:
        assert isinstance(p, IdentityOperator)
    elif n == 1:
        assert p is op
    else:
        assert isinstance(p, CompositionOperator)
        for o in p.operands:
            assert o is op


@pytest.mark.parametrize('n', [-1.2, -1, -0.5, 0, 0.5, 1, 2.4])
def test_pow3(n):
    diag = np.array([1.0, 2, 3])
    d = DiagonalOperator(diag)
    assert_equal((d**n).todense(), DiagonalOperator(diag**n).todense())


# ========================
# Test input/output shape
# ========================


@pytest.mark.parametrize(
    'shape',
    [
        3,
        [3],
        np.array(3),
        np.array([3]),
        (3,),
        3.0,
        [3.0],
        np.array(3.0),
        np.array([3.0]),
        (3.0,),
        [3, 2],
        np.array([3, 2]),
        (3, 2),
        [3.0, 2],
        np.array([3.0, 2]),
        (3.0, 2),
    ],
)
def test_shape_is_inttuple(shape):
    op = Operator(shapein=shape, shapeout=shape)
    assert_is_inttuple(op.shapein)
    assert_is_inttuple(op.shapeout)


def test_shape_explicit():

    o1, o2, o3 = (
        Operator(shapeout=(13, 2), shapein=(2, 2), flags='linear'),
        Operator(shapeout=(2, 2), shapein=(1, 3), flags='linear'),
        Operator(shapeout=(1, 3), shapein=4, flags='linear'),
    )

    for op, eout, ein in zip(
        [o1 * o2, o2 * o3, o1 * o2 * o3],
        ((13, 2), (2, 2), (13, 2)),
        ((1, 3), (4,), (4,)),
    ):
        assert_equal(op.shapeout, eout)
        assert_equal(op.shapein, ein)

    with pytest.raises(ValueError):
        CompositionOperator([o2, o1])
    with pytest.raises(ValueError):
        CompositionOperator([o3, o2])
    with pytest.raises(ValueError):
        CompositionOperator([o3, I, o1])

    o4 = Operator(shapeout=o1.shapeout)
    o5 = Operator(flags='square')

    o1 = Operator(shapein=(13, 2), flags='square')
    for op in [o1 + I, I + o1, o1 + o4, o1 + I + o5 + o4, I + o5 + o1]:
        assert_equal(op.shapeout, o1.shapeout)
        assert_equal(op.shapein, o1.shapein)

    with pytest.raises(ValueError):
        AdditionOperator([o2, o1])
    with pytest.raises(ValueError):
        AdditionOperator([o3, o2])
    with pytest.raises(ValueError):
        AdditionOperator([I, o3, o1])
    with pytest.raises(ValueError):
        AdditionOperator([o3, I, o1])


@flags.linear
class OpShapeImplicit(Operator):
    def __init__(self, factor):
        self.factor = factor
        Operator.__init__(self)

    def reshapein(self, shape):
        return shape[0] * self.factor

    def reshapeout(self, shape):
        return shape[0] / self.factor

    def __str__(self):
        return super().__str__() + f'x{self.factor}'


@pytest.mark.parametrize(
    'op', [OpShapeImplicit(2), OpShapeImplicit(3), OpShapeImplicit(4)]
)
def test_shape_implicit1(op):
    assert op.shapein is None
    assert op.shapeout is None


@pytest.mark.parametrize(
    'op, eout, ein',
    [
        (OpShapeImplicit(2) * OpShapeImplicit(3), (6,), (4,)),
        (OpShapeImplicit(3) * OpShapeImplicit(4), (12,), (2,)),
        (OpShapeImplicit(2) * OpShapeImplicit(3) * OpShapeImplicit(4), (24,), (1,)),
    ],
)
def test_shape_implicit2(op, eout, ein):
    assert_equal(op.reshapein((1,)), eout)
    assert_equal(op.reshapeout((24,)), ein)


@pytest.mark.parametrize('shape', SHAPES)
def test_shapeout_unconstrained1(shape):
    op = Operator(shapein=shape)
    assert op.shapeout is None


@pytest.mark.parametrize('s1', SHAPES)
@pytest.mark.parametrize('s2', SHAPES)
def test_shapeout_unconstrained2(s1, s2):
    @flags.linear
    class Op(Operator):
        def direct(self, input, output):
            output[...] = 4

    op = IdentityOperator(shapein=s1) * Op(shapein=s2)
    if s1 is not None:
        assert op.shapeout == s1
    else:
        assert op.shapeout is None


@pytest.mark.parametrize('shapein', SHAPES)
def test_shapeout_implicit(shapein):
    class Op(Operator):
        def reshapein(self, shape):
            return shape + (2,)

    with pytest.raises(ValueError):
        Op(shapein=3, shapeout=11)

    op = Op(shapein=shapein)
    assert not op.flags.square
    assert op.shapein == shapein
    if shapein is None:
        assert op.shapeout is None
    else:
        assert op.shapeout == shapein + (2,)


@pytest.mark.parametrize('shape', SHAPES[1:])
def test_shapein_unconstrained1(shape):
    op = Operator(shapeout=shape)
    assert op.shapein is None


@pytest.mark.parametrize('shapeout', SHAPES[1:])
def test_shapein_unconstrained2(shapeout):
    class Op(Operator):
        def reshapeout(self, shape):
            return shape + (2,)

    with pytest.raises(ValueError):
        Op(shapein=3, shapeout=11)

    op = Op(shapeout=shapeout)
    assert not op.flags.square
    assert op.shapeout == shapeout
    assert op.shapein == shapeout + (2,)


@flags.square
class OpShapeinUnconstrained1(Operator):
    pass


@flags.square
class OpShapeinUnconstrained2(Operator):
    def reshapein(self, shape):
        return shape

    def toshapein(self, v):
        return v


@flags.square
class OpShapeinUnconstrained3(Operator):
    def reshapeout(self, shape):
        return shape

    def toshapeout(self, v):
        return v


@flags.square
class OpShapeinUnconstrained4(Operator):
    def reshapein(self, shape):
        return shape

    def reshapeout(self, shape):
        return shape

    def toshapein(self, v):
        return v

    def toshapeout(self, v):
        return v


@pytest.mark.parametrize(
    'cls',
    [
        OpShapeinUnconstrained1,
        OpShapeinUnconstrained2,
        OpShapeinUnconstrained3,
        OpShapeinUnconstrained4,
    ],
)
@pytest.mark.parametrize('shapeout', SHAPES[1:])
def test_shapein_unconstrained3(cls, shapeout):
    op = cls(shapeout=shapeout)
    assert_square(op)
    assert_equal(op.shapein, shapeout)


# ================
# Test validation
# ================


@pytest.mark.parametrize('cls', OPS)
def test_validation(cls):
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

    op = cls(validatein=vin, validateout=vout)
    op(x_ok, y_ok)
    cls_error = ValueError if op.flags.shape_input == 'explicit' else ValidationError
    with pytest.raises(cls_error):
        op(x_err, y_ok)

    cls_error = ValueError if op.flags.shape_output == 'explicit' else ValidationError
    with pytest.raises(cls_error):
        op(x_ok, y_err)

    if op.flags.shape_output == 'implicit':
        with pytest.raises(ValidationError):
            cls(validateout=vout, shapein=x_err.shape)
    if op.flags.shape_input == 'implicit':
        with pytest.raises(ValidationError):
            cls(validatein=vin, shapeout=y_err.shape)


# ====================
# Test operator dtype
# ====================


@pytest.mark.parametrize('dop', DTYPES)
@pytest.mark.parametrize('di', DTYPES)
def test_dtype1(dop, di):
    @flags.square
    class Op(Operator):
        def __init__(self, dtype):
            Operator.__init__(self, dtype=dtype)

        def direct(self, input, output):
            np.multiply(input, np.array(value, self.dtype), output)

    value = 2.5
    input = complex(1, 1)
    try:
        i = np.array(input, di)
    except TypeError:
        i = np.array(input.real, di)
    o = Op(dop)(i)
    assert_equal(o.dtype, (i * np.array(value, dop)).dtype, str((dop, di)))
    assert_equal(o, i * np.array(value, dop), str((dop, di)))


@pytest.mark.parametrize('di', DTYPES)
def test_dtype2(di):
    @flags.linear
    @flags.square
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)

    op = Op()
    input = complex(1, 1)

    try:
        i = np.array(input, di)
    except TypeError:
        i = np.array(input.real, di)
    o = op(i)
    assert_equal(o.dtype, (i * i).dtype, str(di))
    assert_equal(o, i * i, str(di))


# ===================
# Test operator name
# ===================


def OpName1():
    class sqrt(Operator):
        pass

    return sqrt()


class OpName2(Operator):
    __name__ = 'sqrt'


@pytest.mark.parametrize(
    'op, name',
    [
        (Operator(), 'Operator'),
        (OpName1(), 'sqrt'),
        (Operator(direct=np.sqrt), 'sqrt'),
        (Operator(name='sqrt'), 'sqrt'),
        (OpName2(), 'sqrt'),
    ],
)
def test_name(op, name):
    assert op.__name__ == name


@flags.linear
class OpMergeName1(Operator):
    def __init__(self, **keywords):
        Operator.__init__(self, **keywords)
        self.set_rule(
            ('.', HomothetyOperator), lambda s, o: s.copy(), CompositionOperator
        )


def OpMergeName2():
    class sqrt(OpMergeName1):
        pass

    return sqrt()


class OpMergeName3(OpMergeName1):
    __name__ = 'sqrt'


@pytest.mark.parametrize(
    'op', [OpMergeName1(name='sqrt'), OpMergeName2(), OpMergeName3()]
)
@pytest.mark.parametrize('h', [I, HomothetyOperator(2)])
def test_merge_name(op, h):
    assert op(h).__name__ == 'sqrt'
    assert h(op).__name__ == 'sqrt'


# =========================
# Test operator comparison
# =========================


@pytest.mark.parametrize('cls', OPS)
def test_eq(cls):
    assert_equal(cls(), cls())


# ================
# Test iadd, imul
# ================


@pytest.mark.parametrize('operation', [operator.iadd, operator.imul])
@pytest.mark.parametrize('cls1', OPS)
@pytest.mark.parametrize('cls2', OPS)
def test_iadd_imul(operation, cls1, cls2):
    op1 = cls1()
    op2 = cls2()
    if operation is operator.iadd:
        op = op1 + op2
        op1 += op2
    else:
        op = op1 * op2.T
        op1 *= op2.T
    assert_equal(op1, op)


# ====================
# Test associativity
# ====================


class OpAssociativity1(Operator):
    pass


class OpAssociativity2(Operator):
    pass


class OpAssociativity3(Operator):
    pass


class OpAssociativity4(Operator):
    pass


@pytest.mark.parametrize(
    'operation', [AdditionOperator, MultiplicationOperator, CompositionOperator]
)
def test_associativity_composite_and_operator(operation):
    def func():
        assert isinstance(op, operation)
        assert_equal(len(op.operands), 3)
        if all(
            isinstance(o, c)
            for o, c in zip(
                op.operands, [OpAssociativity2, OpAssociativity3, OpAssociativity1]
            )
        ):
            pytest.xfail('Unknown.')  # commutative rules do not preserve order...
        for o, c in zip(
            op.operands, [OpAssociativity1, OpAssociativity2, OpAssociativity3]
        ):
            assert isinstance(o, c)

    for operation in (AdditionOperator, MultiplicationOperator, CompositionOperator):
        op = operation(
            [operation([OpAssociativity1(), OpAssociativity2()]), OpAssociativity3()]
        )
        func()
        op = operation(
            [OpAssociativity1(), operation([OpAssociativity2(), OpAssociativity3()])]
        )
        func()


@pytest.mark.parametrize(
    'operation', [AdditionOperator, MultiplicationOperator, CompositionOperator]
)
def test_associativity_composite_and_composite(operation):

    op = operation(
        [
            operation([OpAssociativity1(), OpAssociativity2()]),
            operation([OpAssociativity3(), OpAssociativity4()]),
        ]
    )
    assert isinstance(op, operation)
    assert len(op.operands) == 4


def test_associativity3():
    a = GroupOperator([OpAssociativity1(), OpAssociativity2()])
    b = GroupOperator([OpAssociativity3(), OpAssociativity4()])

    for o1, o2 in [(OpAssociativity1(), a), (a, OpAssociativity1()), (a, b)]:
        op = o1(o2)
        assert isinstance(op, CompositionOperator)
        assert_equal(len(op.operands), 2)
        assert op.operands[0] is o1
        assert op.operands[1] is o2


# ================
# Test composite
# ================

OPERANDS_COMPOSITE = [
    Operator(shapein=2, flags='square'),
    Operator(shapein=2, flags='square'),
    Operator(shapein=2, flags='square'),
]


@pytest.mark.parametrize(
    'cls',
    [
        AdditionOperator,
        BlockColumnOperator,
        BlockDiagonalOperator,
        BlockRowOperator,
        BlockSliceOperator,
        CompositionOperator,
        GroupOperator,
        MultiplicationOperator,
    ],
)
@pytest.mark.parametrize(
    'func',
    [
        lambda: list(OPERANDS_COMPOSITE),
        lambda: tuple(OPERANDS_COMPOSITE),
        lambda: (_ for _ in OPERANDS_COMPOSITE),
    ],
)
def test_composite(cls, func):
    operands = func()
    if cls is BlockColumnOperator:
        op = cls(operands, axisout=0)
    elif cls in (BlockDiagonalOperator, BlockRowOperator):
        op = cls(operands, axisin=0)
    elif cls is BlockSliceOperator:
        op = cls(operands, (slice(i, i + 2) for i in (0, 2, 4)))
    else:
        op = cls(operands)
    assert type(op.operands) is list


# ==================
# Test commutative
# ==================


def test_addition():
    @flags.square
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)

        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.sum([Op(v) for v in [1]])
    assert op.__class__ is Op

    op = np.sum([Op(v) for v in [1, 2]])
    assert_equal(op.__class__, AdditionOperator)

    pool.clear()
    assert_equal(op(1), 3)
    assert_equal(len(pool), 1)

    op = np.sum([Op(v) for v in [1, 2, 4]])
    assert op.__class__ is AdditionOperator

    pool.clear()
    input = np.array(1, int)
    output = np.array(0, int)
    assert_equal(op(input, output), 7)
    assert_equal(input, 1)
    assert_equal(output, 7)
    assert_equal(len(pool), 1)

    pool.clear()
    output = input
    assert_equal(op(input, output), 7)
    assert_equal(input, 7)
    assert_equal(output, 7)
    assert_equal(len(pool), 2)


@pytest.mark.parametrize(
    'flag', ['linear', 'real', 'square', 'symmetric', 'hermitian', 'separable']
)
def test_addition_flags(flag):
    o = AdditionOperator([Operator(flags=flag), Operator(flags=flag)])
    assert getattr(o.flags, flag)


def test_multiplication():
    @flags.square
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)

        def direct(self, input, output):
            np.multiply(input, self.v, output)

    pool.clear()
    op = MultiplicationOperator([Op(v) for v in [1]])
    assert op.__class__ is Op

    op = MultiplicationOperator([Op(v) for v in [1, 2]])
    assert_equal(op.__class__, MultiplicationOperator)
    assert_equal(op(1), 2)
    assert_equal(len(pool), 1)

    op = MultiplicationOperator([Op(v) for v in [1, 2, 4]])
    assert op.__class__ is MultiplicationOperator

    input = np.array(1, int)
    output = np.array(0, int)
    assert_equal(op(input, output), 8)
    assert_equal(input, 1)
    assert_equal(output, 8)
    assert_equal(len(pool), 1)

    output = input
    assert_equal(op(input, output), 8)
    assert_equal(input, 8)
    assert_equal(output, 8)
    assert_equal(len(pool), 2)


@pytest.mark.parametrize('flag', ['real', 'square', 'separable'])
def test_multiplication_flags(flag):
    o = MultiplicationOperator([Operator(flags=flag), Operator(flags=flag)])
    assert getattr(o.flags, flag)


@pytest.mark.parametrize('cls', [AdditionOperator, MultiplicationOperator])
@pytest.mark.parametrize('OP1', OPS)
@pytest.mark.parametrize('OP2', OPS)
def test_commutative_shapes(cls, OP1, OP2):
    n1 = OP1.__name__
    n2 = OP2.__name__
    op = cls([OP1(), OP2()])

    shape_output = op.flags.shape_output
    if 'Expl' in (n1[:4], n2[:4]):
        assert shape_output == 'explicit'
    elif n1[4:] == 'Expl' and n2[:4] == 'Impl' or n2[4:] == 'Expl' and n1[:4] == 'Impl':
        assert shape_output == 'explicit'
    elif 'Impl' in (n1[:4], n2[:4]):
        assert shape_output == 'implicit'
    else:
        assert shape_output == 'unconstrained'

    shape_input = op.flags.shape_input
    if 'Expl' in (n1[4:], n2[4:]):
        assert shape_input == 'explicit'
    elif n1[:4] == 'Expl' and n2[4:] == 'Impl' or n2[:4] == 'Expl' and n1[4:] == 'Impl':
        assert shape_input == 'explicit'
    elif 'Impl' in (n1[4:], n2[4:]):
        assert shape_input == 'implicit'
    else:
        assert shape_input == 'unconstrained'


# ==================
# Test Block slice
# ==================


@pytest.mark.parametrize('ndim', range(1, 5))
@pytest.mark.parametrize('nops', range(1, 5))
@pytest.mark.parametrize('cls', [HomothetyOperator, HomothetyOutplace])
def test_block_slice(ndim, nops, cls):
    size = 4
    slices_ = [
        [split(size, nops, i) for i in range(nops)],
        [split(size, size, i) for i in range(nops)],
        [ndim * (slice(i, None, nops),) for i in range(nops)],
    ]

    for slices in slices_:
        input = np.zeros(ndim * (size,))
        expected = np.zeros_like(input)
        ops = [cls(i + 1) for i in range(nops)]
        for i, s in enumerate(slices):
            input[s] = 10 * (i + 1)
            expected[s] = input[s] * (i + 1)
        o = BlockSliceOperator(ops, slices)
        assert o.flags.inplace is cls.flags.inplace
        actual = o(input)
        assert_equal(actual, expected)
        o(input, input)
        assert_equal(input, expected)


def test_block_slice_rule_homothety():
    b = BlockSliceOperator(2 * [HomothetyOperator(3)], [slice(0, 10), slice(12, 14)])
    hb = HomothetyOperator(2) * b
    assert isinstance(hb, BlockSliceOperator)
    for op in hb.operands:
        assert isinstance(op, HomothetyOperator)
        assert_equal(op.data, 6)


# ====================
# Test copy operator
# ====================


def test_copy():
    C = CopyOperator()
    x = np.array([10, 20])
    assert_equal(x, C(x))
    x_ = x.copy()
    C(x, x)
    assert_equal(x, x_)


# ========================
# Test ReductionOperator
# ========================


@pytest.mark.parametrize('func', [np.add, np.multiply, np.min, np.max, np.sum, np.prod])
@pytest.mark.parametrize('shape', SHAPES[2:])
def test_reduction_operator(func, shape):

    for a in [None] + list(range(len(shape))):
        op = ReductionOperator(func, axis=a)
        v = np.arange(product(shape)).reshape(shape)
        if isinstance(func, np.ufunc):
            if np.__version__ < '1.7' and a is None:
                expected = func.reduce(v.flat, 0)
            else:
                expected = func.reduce(v, a)
        else:
            expected = func(v, axis=a)
        assert_equal(op(v), expected)
        out = np.empty_like(expected)
        op(v, out)
        assert_equal(out, expected)


@pytest.mark.parametrize('func', [np.cos, np.modf])
def test_reduction_operator_invalid1(func):
    with pytest.raises(TypeError):
        ReductionOperator(func)


@pytest.mark.parametrize('a', (1, 2, 3))
def test_reduction_operator_invalid2(a):
    op = ReductionOperator(np.add, axis=a)
    for n in range(0, a + 1):
        v = np.empty(n * [2])
        with pytest.raises(TypeError if n == 0 else ValueError):
            op(v)


# =================
# Test asoperator
# =================


@pytest.mark.parametrize('dtype', DTYPES)
def test_asoperator_scalar_homothety(dtype):
    scalar = np.array(1, dtype)
    o = asoperator(scalar)
    assert isinstance(o, HomothetyOperator)


@pytest.mark.parametrize('dtype', DTYPES)
def test_asoperator_scalar_constant(dtype):
    scalar = np.array(1, dtype)
    o = asoperator(scalar, constant=True)
    assert isinstance(o, ConstantOperator)


@pytest.mark.parametrize(
    'cls, values',
    [
        (IdentityOperator, [1]),
        (HomothetyOperator, [2]),
        (DiagonalOperator, [1, 2]),
        (DenseOperator, [[1]]),
        (DenseOperator, [[1, 2]]),
        (DenseOperator, [[1, 2], [2, 3]]),
        (DenseBlockDiagonalOperator, [[[1, 2], [2, 3]]]),
        (
            DenseBlockDiagonalOperator,
            [[[1, 2], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
        ),
    ],
)
@pytest.mark.parametrize('func', [lambda x: x, totuple, np.array])
def test_asoperator_ndarray(cls, values, func):
    values = func(values)
    shape = np.array(values).shape
    if len(shape) > 1:
        expected_shape = shape[:-2] + (shape[-1],)
    else:
        expected_shape = shape

    o = asoperator(values)
    assert isinstance(o, cls)
    assert o.shapein == expected_shape


@pytest.mark.parametrize(
    'values',
    [
        [1],
        [2],
        [1, 2],
        [[1]],
        [[1, 2]],
        [[1, 2], [2, 3]],
        [[[1, 2], [2, 3]]],
        [[[1, 2], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
    ],
)
def test_asoperator_ndarray_constant(values):
    vt = totuple(values)
    va = np.array(values)
    shape = va.shape
    for v in values, vt, va:
        o = asoperator(v, constant=True)
        assert isinstance(o, ConstantOperator)
        assert shape == o.shapeout


@pytest.mark.parametrize('input', [2, [2], [2, 3]])
def test_asoperator_func(input):
    f = lambda x: x**2
    op = asoperator(f)
    assert isinstance(op, Operator)
    assert op.flags.inplace
    assert_equal(op(input), f(np.array(input)))
