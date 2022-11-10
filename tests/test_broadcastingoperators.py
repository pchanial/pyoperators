import itertools
import operator

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import (
    AdditionOperator,
    BlockDiagonalOperator,
    CompositionOperator,
    ConstantOperator,
    DiagonalNumexprOperator,
    DiagonalOperator,
    HomothetyOperator,
    I,
    IdentityOperator,
    MaskOperator,
    MultiplicationOperator,
    O,
    Operator,
    PackOperator,
    UnpackOperator,
    ZeroOperator,
)
from pyoperators.core import BroadcastingBase
from pyoperators.flags import linear, square
from pyoperators.rules import rule_manager
from pyoperators.utils import float_or_complex_dtype, product
from pyoperators.utils.testing import assert_same

from .common import DTYPES, HomothetyOutplace

CLASSES = (
    ConstantOperator,
    DiagonalOperator,
    DiagonalNumexprOperator,
    HomothetyOperator,
    IdentityOperator,
    MaskOperator,
    ZeroOperator,
)
SAMESHAPES = ((False, True), (True, True), (True, True), (True, True))


def get_operator(cls, data, **keywords):
    if cls is DiagonalNumexprOperator:
        args = (data, '3 * data')
    elif cls is HomothetyOperator:
        args = (data.flat[0],)
    elif cls in (IdentityOperator, ZeroOperator):
        args = ()
    else:
        args = (data,)
    return cls(*args, **keywords)


@pytest.mark.parametrize(
    'data, expected',
    [
        (0.0, ZeroOperator),
        (1.0, IdentityOperator),
        ([0, 0], ZeroOperator),
        ([1, 1], IdentityOperator),
        (2, HomothetyOperator),
        ([2, 2], HomothetyOperator),
        ([0, 1], MaskOperator),
        ([-1, -1], HomothetyOperator),
        ([-1, 1], DiagonalOperator),
        ([2, 1], DiagonalOperator),
    ],
)
def test_diagonal1(data, expected):
    op = DiagonalOperator(data)
    if all(_ in (-1, 1) for _ in op.data.flat):
        assert op.flags.involutary
    assert type(op) is expected


TEST_DIAGONAL2_OPS = [
    DiagonalOperator([1.0, 2], broadcast='rightward'),
    DiagonalOperator([[2.0, 3, 4], [5, 6, 7]], broadcast='rightward'),
    DiagonalOperator([1.0, 2, 3, 4, 5], broadcast='leftward'),
    DiagonalOperator(np.arange(20).reshape(4, 5), broadcast='leftward'),
    DiagonalOperator(np.arange(120.0).reshape(2, 3, 4, 5)),
    HomothetyOperator(7.0),
    IdentityOperator(),
]


@pytest.mark.parametrize(
    'composite_cls', (AdditionOperator, CompositionOperator)
)  # MultiplicationOperator
@pytest.mark.parametrize(
    'diagonal_operators', itertools.combinations(TEST_DIAGONAL2_OPS, 2)
)
def test_diagonal2(composite_cls, diagonal_operators):
    d1, d2 = diagonal_operators
    if {d1.broadcast, d2.broadcast} == {'leftward', 'rightward'}:
        return

    x = np.arange(120.0).reshape(2, 3, 4, 5) / 2

    operation = {
        AdditionOperator: operator.add,
        CompositionOperator: operator.mul,
        MultiplicationOperator: operator.mul,
    }[composite_cls]
    op = composite_cls(diagonal_operators)

    if type(d1) is DiagonalOperator:
        assert type(op) is DiagonalOperator
    elif type(d1) is HomothetyOperator:
        assert type(op) is HomothetyOperator
    elif operation is CompositionOperator:
        assert type(op) is IdentityOperator
    else:
        assert type(op) is HomothetyOperator

    data = (
        operation(d1.data.T, d2.data.T).T
        if 'rightward' in (d1.broadcast, d2.broadcast)
        else operation(d1.data, d2.data)
    )
    assert_same(op.data, data)
    if composite_cls is CompositionOperator:
        assert_same(op(x), d1(d2(x)))
    else:
        assert_same(op(x), operation(d1(x), d2(x)))


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

    b = np.array([3.0, 4.0, 1.0, 0.0, 3.0, 2.0])
    c = np.array([3.0, 4.0, 0.0, 0.0, 3.0, 0.0])
    mask = MaskOperator(np.array([0, 0.0, 1.0, 1.0, 0.0, 1], dtype=np.int8))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([1, 1.0, 0.0, 0.0, 1.0, 0]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([False, False, True, True, False, True]))
    assert np.all(mask(b) == c)

    b = np.array([[3.0, 4.0], [1.0, 0.0], [3.0, 2.0]])
    c = np.array([[3.0, 4.0], [0.0, 0.0], [3.0, 0.0]])
    mask = MaskOperator(np.array([[0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype='int8'))
    assert np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([[1, 1.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(np.array([[False, False], [True, True], [False, True]]))
    assert np.all(mask(b) == c)

    b = np.array([[[3, 4.0], [1.0, 0.0]], [[3.0, 2], [-1, 9]]])
    c = np.array([[[3, 4.0], [0.0, 0.0]], [[3.0, 0], [0, 0]]])
    mask = MaskOperator(np.array([[[0, 0.0], [1.0, 1.0]], [[0.0, 1], [1, 1]]], int))
    assert np.all(mask(b) == c)

    mask = DiagonalOperator(np.array([[[1, 1], [0.0, 0]], [[1, 0], [0, 0]]]))
    assert np.all(mask(b) == c)
    mask = MaskOperator(
        np.array([[[False, False], [True, True]], [[False, True], [True, True]]])
    )
    assert np.all(mask(b) == c)

    c = mask(b, b)
    assert id(b) == id(c)


def test_masking2():
    m = MaskOperator([True, False, True])
    assert m @ m == m


def test_homothety_operator_one():
    s = HomothetyOperator(1)
    assert s.C is s.T is s.H is s.I is s


def test_homothety_operator_minus_one():
    s = HomothetyOperator(-1)
    assert s.C is s.T is s.H is s.I is s


def test_homothety_operator_real():
    s = HomothetyOperator(2.0)
    assert s.C is s.T is s.H is s
    assert s.I is not s

    for op in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        assert isinstance(op, HomothetyOperator)


def test_homothety_operator_complex():
    s = HomothetyOperator(complex(1, 1))
    assert s.T is s
    assert s.H is s.C
    assert s.I not in (s, s.C)
    assert s.I.C not in (s, s.C)

    for op in (s.C, s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        assert isinstance(op, HomothetyOperator)


@pytest.mark.parametrize(
    'model, expected',
    [
        (1.0 * I + I, 6),
        (-I, -3),
        ((-2) * I, -6),
        (-(2 * I), -6),
        (1.0 * I - I, 0),
        (1.0 * I - 2 * I, -3),
    ],
)
@pytest.mark.parametrize('input', [np.array(3), [3], (3,), np.int64(3), 3])
def test_homothety_rules1(model, input, expected):
    output = model(input)
    assert_equal(output, expected, str((model, input)))
    assert output.dtype == int


@pytest.mark.parametrize(
    'ioperation, model1, model2, expected',
    [
        (operator.iadd, -I, 2 * I, 3),
        (operator.isub, I, 2 * I, -3),
        (operator.imul, -I, 2 * I, -6),
        (operator.iadd, -2 * I, O, -6),
        (operator.imul, -2 * I, O, 0),
    ],
)
@pytest.mark.parametrize('input', [np.array(3), [3], (3,), np.int64(3), 3])
def test_homothety_rules2(input, ioperation, model1, model2, expected):
    model = ioperation(model1, model2)
    assert_equal(model(input), expected)


@pytest.mark.parametrize('opout', [None, (100,)])
@pytest.mark.parametrize('opin', [None, (100,)])
@pytest.mark.parametrize('idin', [None, (100,)])
def test_homothety_rules3(opout, opin, idin):
    @linear
    class Op(Operator):
        pass

    if opin is not None and idin is not None and opin != idin:
        return
    p = Op(shapeout=opout, shapein=opin) @ IdentityOperator(shapein=idin)

    if idin is None:
        idin = opin
    assert isinstance(p, Op)
    assert p.shapein == idin
    assert p.shapeout == opout


@pytest.mark.parametrize(
    'value1, broadcast1',
    [
        (1, 'scalar'),
        (np.array([1, 2]), 'rightward'),
        (np.array([2, 3, 4]), 'leftward'),
    ],
)
@pytest.mark.parametrize(
    'value2, broadcast2',
    [
        (1, 'scalar'),
        (np.array([1, 2]), 'rightward'),
        (np.array([2, 3, 4]), 'leftward'),
    ],
)
def test_constant_rules1(value1, broadcast1, value2, broadcast2):
    op1 = ConstantOperator(value1, broadcast=broadcast1)
    op2 = ConstantOperator(value2, broadcast=broadcast2)
    op = op1 + op2

    if {op1.broadcast, op2.broadcast} != {'rightward', 'leftward'}:
        assert isinstance(op, ConstantOperator)
    v = np.zeros((2, 3))
    op(np.nan, v)
    z = np.zeros((2, 3))
    if broadcast1 == 'rightward':
        z.T[...] += value1.T
    else:
        z[...] += value1
    if broadcast2 == 'rightward':
        z.T[...] += value2.T
    else:
        z[...] += value2
    assert_equal(v, z)


OPERATOR_CONSTANT_RULES2 = HomothetyOperator(2, shapein=(2, 3)) @ Operator(
    direct=np.square, shapein=(2, 3), flags='linear,square'
)


@pytest.mark.parametrize(
    'cop, oop, expected_type, expected_data',
    [
        (ConstantOperator(3), I, HomothetyOperator, 3),
        (
            ConstantOperator(3),
            OPERATOR_CONSTANT_RULES2,
            CompositionOperator,
            (HomothetyOperator, 6),
        ),
        (ConstantOperator(3), HomothetyOperator(5), HomothetyOperator, 15),
        (
            ConstantOperator([1, 2, 3], broadcast='leftward'),
            I,
            DiagonalOperator,
            [1, 2, 3],
        ),
        (
            ConstantOperator([1, 2, 3], broadcast='leftward'),
            OPERATOR_CONSTANT_RULES2,
            CompositionOperator,
            (DiagonalOperator, [2, 4, 6]),
        ),
        (
            ConstantOperator([1, 2, 3], broadcast='leftward'),
            HomothetyOperator(5),
            DiagonalOperator,
            [5, 10, 15],
        ),
        (ConstantOperator(np.ones((2, 3))), I, IdentityOperator, 1),
        (
            ConstantOperator(np.ones((2, 3))),
            OPERATOR_CONSTANT_RULES2,
            CompositionOperator,
            (HomothetyOperator, 2),
        ),
        (ConstantOperator(np.ones((2, 3))), HomothetyOperator(5), HomothetyOperator, 5),
    ],
)
def test_constant_rules2(cop, oop, expected_type, expected_data):
    v = np.arange(6).reshape((2, 3))
    op = MultiplicationOperator([cop, oop])
    assert type(op) is expected_type
    assert_equal(op(v), cop.data * oop(v))

    if type(op) is CompositionOperator:
        op = op.operands[0]
        expected_type, expected_data = expected_data
        assert type(op) is expected_type

    assert_equal(op.data, expected_data)


@pytest.mark.parametrize(
    'shapein, constant_op',
    [
        ((), ConstantOperator(2)),
        (4, ConstantOperator([2], broadcast='leftward')),
        (
            (2, 3, 4),
            ConstantOperator(2 * np.arange(8).reshape((2, 1, 4)), broadcast='leftward'),
        ),
    ],
)
def test_constant_rules3(shapein, constant_op):
    @square
    class Op(Operator):
        def direct(self, input, output):
            output[...] = input + np.arange(input.size).reshape(input.shape)

    op = Op(shapein=shapein)
    v = 10000000

    y_tmp = np.empty(shapein, int)
    constant_op(v, y_tmp)
    assert_equal((op @ constant_op)(v), op(y_tmp))


@pytest.mark.parametrize(
    'valid, x',
    [
        (np.array([False, True, True, False, True, True]), np.arange(6)),
        (
            np.array([[False, True, True], [False, True, True]]),
            np.arange(6).reshape(2, 3),
        ),
    ],
)
@pytest.mark.parametrize('shape', [(), (4,), (4, 5)])
@pytest.mark.parametrize('broadcast', ['disabled', 'leftward', 'rightward'])
def test_packing(valid, x, shape, broadcast):
    expected = np.array([1, 2, 4, 5])
    p = PackOperator(valid, broadcast=broadcast)
    masking = MaskOperator(~valid, broadcast=broadcast)
    if broadcast == 'leftward':
        x_ = np.empty(shape + x.shape)
        x_[...] = x
        expected_ = np.empty(shape + (expected.size,))
        expected_[...] = expected
    else:
        x_ = np.empty(x.shape + shape)
        x_.reshape((x.size, -1))[...] = x.ravel()[..., None]
        expected_ = np.empty((expected.size,) + shape)
        expected_.reshape((expected.size, -1))[...] = expected[..., None]

    if broadcast == 'disabled' and shape != ():
        with pytest.raises(ValueError):
            p(x_)
        return
    assert_equal(p(x_), expected_)

    assert type(p.T) is UnpackOperator
    assert p.T.broadcast == p.broadcast
    assert_equal(p.T(expected_), masking(x_))

    u = UnpackOperator(valid, broadcast=broadcast)
    assert type(u.T) is PackOperator
    assert u.T.broadcast == u.broadcast
    assert_equal(u(expected_), masking(x_))
    assert_equal(u.T(x_), expected_)


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('dtype', [bool] + DTYPES)
def test_dtype(cls, dtype):
    x = np.array([3, 0, 2])
    op = get_operator(cls, x.astype(dtype))
    c_ = type(op)
    if c_ in (IdentityOperator, ZeroOperator):
        expected_dtype = int
    elif c_ is MaskOperator:
        expected_dtype = bool
    else:
        expected_dtype = dtype
    assert op.data.dtype == expected_dtype

    if c_ in (IdentityOperator, MaskOperator, ZeroOperator):
        assert op.dtype == None
    elif c_ is DiagonalNumexprOperator:
        assert op.dtype == float_or_complex_dtype(dtype)
    else:
        assert op.dtype == dtype


@pytest.mark.parametrize('cls, sameshapes', zip(CLASSES, SAMESHAPES))
@pytest.mark.parametrize('shape', [(), (3,), (3, 2)])
@pytest.mark.parametrize('broadcast', [None, 'leftward', 'rightward'])
def test_shape(cls, sameshapes, shape, broadcast):
    sameshapein, sameshapeout = sameshapes

    x = np.arange(product(shape)).reshape(shape)
    op = get_operator(cls, x, broadcast=broadcast)
    if len(shape) == 0 or cls in (HomothetyOperator, IdentityOperator, ZeroOperator):
        assert op.broadcast == 'scalar'
        assert op.shapein is None
        assert op.shapeout is None
    elif broadcast in ('leftward', 'rightward'):
        assert op.broadcast == broadcast
        assert op.shapein is None
        assert op.shapeout is None
    else:
        assert op.broadcast == 'disabled'
        if sameshapein:
            assert op.shapein == shape
        if sameshapeout:
            assert op.shapeout == shape


@pytest.mark.parametrize(
    'cls, commutative, left, right',
    [
        (ConstantOperator, True, False, False),
        (DiagonalOperator, True, True, True),
        (DiagonalNumexprOperator, True, True, True),
        (HomothetyOperator, True, True, True),
        (IdentityOperator, True, True, True),
        (MaskOperator, True, True, True),
        (PackOperator, True, False, True),
        (UnpackOperator, True, True, False),
    ],
)
@pytest.mark.parametrize('ndim', range(3))
def test_partition(cls, commutative, left, right, ndim):
    def assert_partition(a, b, operation, apply_rule):
        p = operation([a, b])
        if not apply_rule:
            if isinstance(a, IdentityOperator) or isinstance(b, IdentityOperator):
                return
            assert not isinstance(p, BlockDiagonalOperator)
            return
        assert isinstance(p, BlockDiagonalOperator)
        with rule_manager(none=True):
            q = operation([a, b])
        assert_equal(p.todense(), q.todense())

    def sfunc1(ndim_):
        s = list(range(2, ndim_ + 2))
        data = np.arange(product(s)).reshape(s) + 2
        if cls is MaskOperator:
            data = (data % 2).astype(bool)
        return data

    def sfunc2(ndim_):
        s = list(range(2 + ndim - ndim_, 2 + ndim))
        data = np.arange(product(s)).reshape(s) + 2
        if cls is MaskOperator:
            data = (data % 2).astype(bool)
        return data

    def toone(index):
        list_ = list(shape)
        list_[index] = 1
        return list_

    def remove(index):
        list_ = list(shape)
        list_.pop(index)
        return list_

    shape = tuple(range(2, 2 + ndim))

    if cls in (HomothetyOperator, IdentityOperator):
        ops = [get_operator(cls, np.array(2))]
    else:
        ops = (
            [get_operator(cls, sfunc1(ndim)) for ndim in range(ndim + 1)]
            + [
                get_operator(cls, sfunc2(ndim), broadcast='leftward')
                for ndim in range(1, ndim + 1)
            ]
            + [
                get_operator(cls, sfunc1(ndim), broadcast='rightward')
                for ndim in range(1, ndim + 1)
            ]
        )

    block = [
        BlockDiagonalOperator(
            [
                HomothetyOutplace(v, shapein=toone(axis))
                for v in range(2, 2 + shape[axis])
            ],
            axisin=axis,
            partitionin=shape[axis] * [1],
        )
        for axis in range(-ndim, ndim)
    ] + [
        BlockDiagonalOperator(
            [
                HomothetyOutplace(v, shapein=remove(axis))
                for v in range(2, 2 + shape[axis])
            ],
            new_axisin=axis,
            partitionin=shape[axis] * [1],
        )
        for axis in range(-ndim, ndim)
    ]

    for o, b in itertools.product(ops, block):
        if (o.shapein is None or o.shapein == b.shapein) and (
            o.shapeout is None or o.shapeout == b.shapeout
        ):
            assert_partition(o, b, AdditionOperator, commutative)
            assert_partition(o, b, MultiplicationOperator, commutative)
        if o.shapein is None or o.shapein == b.shapeout:
            assert_partition(o, b, CompositionOperator, right)
        if o.shapeout is None or b.shapein == o.shapeout:
            assert_partition(b, o, CompositionOperator, left)


@pytest.mark.parametrize(
    'broadcast, shape',
    [
        ('leftward', (2, 4, 3, 4, 2, 2)),
        ('rightward', (3, 2, 2, 3, 1, 2)),
    ],
)
def test_as_strided(broadcast, shape):
    op = BroadcastingBase(np.arange(6).reshape((3, 1, 2, 1)), broadcast)
    if broadcast == 'leftward':
        v = op.data * np.ones(shape)
    else:
        v = (op.data.T * np.ones(shape, int).T).T
    assert_equal(op._as_strided(shape), v)
