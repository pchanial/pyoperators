import numpy as np
import pytest

from pyoperators import I, Operator, flags
from pyoperators.utils import ndarraywrap


class ndarray2(np.ndarray):
    pass


class ndarray3(np.ndarray):
    pass


class ndarray4(np.ndarray):
    pass


@flags.linear
@flags.square
class Op2(Operator):
    attrout = {'newattr': True}

    def direct(self, input, output):
        pass

    def transpose(self, input, output):
        pass


@flags.linear
@flags.square
class Op3(Operator):
    classout = ndarray3
    classin = ndarray4

    def direct(self, input, output):
        pass

    def transpose(self, input, output):
        pass


@flags.linear
@flags.square
class O1(Operator):
    classout = ndarray2

    def direct(self, input, output):
        output[...] = input


@flags.linear
@flags.square
class O2(Operator):
    def direct(self, input, output):
        output[...] = input


def assert_direct(op, i, c):
    o = op(i)
    assert type(o) is c


def assert_direct_inplace(op, i, c):
    i = i.copy()
    op(i, i)
    assert type(i) is c


@pytest.mark.parametrize(
    'op, input, expected_class',
    [
        (I, np.ones(2), np.ndarray),
        (Op2(), np.ones(2), ndarraywrap),
        (Op2() @ Op3(), np.ones(2), ndarray3),
        (Op3() @ Op2(), np.ones(2), ndarray3),
        (I, np.ones(2).view(ndarray2), ndarray2),
        (Op2(), np.ones(2).view(ndarray2), ndarray2),
        (Op2() @ Op3(), np.ones(2).view(ndarray2), ndarray3),
        (Op3() @ Op2(), np.ones(2).view(ndarray2), ndarray3),
    ],
)
def test_propagation_class(op, input, expected_class):
    assert_direct(op, input, expected_class)


@pytest.mark.parametrize(
    'op, input, expected_class',
    [
        (I, np.ones(2), np.ndarray),
        (Op2(), np.ones(2), np.ndarray),
        (Op2() @ Op3(), np.ones(2), np.ndarray),
        (Op3() @ Op2(), np.ones(2), np.ndarray),
        (I, np.ones(2).view(ndarray2), ndarray2),
        (Op2(), np.ones(2).view(ndarray2), ndarray2),
        (Op2() @ Op3(), np.ones(2).view(ndarray2), ndarray3),
        (Op3() @ Op2(), np.ones(2).view(ndarray2), ndarray3),
        (I, np.ones(2).view(ndarray3), ndarray3),
        (Op2(), np.ones(2).view(ndarray3), ndarray3),
        (Op2() @ Op3(), np.ones(2).view(ndarray3), ndarray3),
        (Op3() @ Op2(), np.ones(2).view(ndarray3), ndarray3),
    ],
)
def test_propagation_class_inplace(op, input, expected_class):
    assert_direct_inplace(op, input, expected_class)


@pytest.mark.parametrize(
    'op, input, expected_class',
    [
        (I, np.ones(2), np.ndarray),
        (Op2(), np.ones(2), np.ndarray),
        (Op2() @ Op3(), np.ones(2), ndarray4),
        (Op3() @ Op2(), np.ones(2), ndarray4),
        (I, np.ones(2).view(ndarray2), ndarray2),
        (Op2(), np.ones(2).view(ndarray2), ndarray2),
        (Op2() @ Op3(), np.ones(2).view(ndarray2), ndarray4),
        (Op3() @ Op2(), np.ones(2).view(ndarray2), ndarray4),
    ],
)
def test_propagation_classT(op, input, expected_class):
    assert_direct(op.T, input, expected_class)


@pytest.mark.parametrize(
    'op, input, expected_class',
    [
        (I, np.ones(2), np.ndarray),
        (Op2(), np.ones(2), np.ndarray),
        (Op2() @ Op3(), np.ones(2), np.ndarray),
        (Op3() @ Op2(), np.ones(2), np.ndarray),
        (I, np.ones(2).view(ndarray2), ndarray2),
        (Op2(), np.ones(2).view(ndarray2), ndarray2),
        (Op2() @ Op3(), np.ones(2).view(ndarray2), ndarray4),
        (Op3() @ Op2(), np.ones(2).view(ndarray2), ndarray4),
        (I, np.ones(2).view(ndarray4), ndarray4),
        (Op2(), np.ones(2).view(ndarray4), ndarray4),
        (Op2() @ Op3(), np.ones(2).view(ndarray4), ndarray4),
        (Op3() @ Op2(), np.ones(2).view(ndarray4), ndarray4),
    ],
)
def test_propagation_classT_inplace(op, input, expected_class):
    assert_direct_inplace(op.T, input, expected_class)


OPS1_PROPAGATION_CLASS_NESTED = [I, 2 * I, O2(), 2 * I + O2()]
OPS2_PROPAGATION_CLASS_NESTED = [
    I + O1(),
    2 * O1(),
    O1() + O2(),
    O2() + O1(),
    I + 2 * O1(),
    I + O1() + O2(),
    I + O2() + O1(),
    O1() + I + O2(),
    O1() + O2() + I,
    O2() + O1() + I,
    O2() + I + O1(),
]


@pytest.mark.parametrize('op1', OPS1_PROPAGATION_CLASS_NESTED)
@pytest.mark.parametrize('op2', OPS2_PROPAGATION_CLASS_NESTED)
def test_propagation_class_2ops_nested(op1, op2):
    assert type((op1 @ op2)(1)) is ndarray2
    assert type((op2 @ op1)(1)) is ndarray2


@pytest.mark.parametrize('op1', OPS1_PROPAGATION_CLASS_NESTED)
@pytest.mark.parametrize('op2', OPS2_PROPAGATION_CLASS_NESTED)
@pytest.mark.parametrize('op3', OPS1_PROPAGATION_CLASS_NESTED)
def test_propagation_class_3ops_nested(op1, op2, op3):
    assert type((op1 @ op2 @ op3)(1)) is ndarray2
