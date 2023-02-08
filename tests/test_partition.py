import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    DiagonalOperator,
    HomothetyOperator,
    I,
    IdentityOperator,
    MultiplicationOperator,
    Operator,
    asoperator,
    flags,
)
from pyoperators.core import BlockOperator
from pyoperators.utils import merge_none

from .common import Stretch


@pytest.fixture(scope='module')
def ref_partition1():
    return DiagonalOperator([1, 2, 2, 3, 3, 3]).todense()


@pytest.mark.parametrize(
    'ops, partition',
    [
        (
            (
                HomothetyOperator(1, shapein=1),
                HomothetyOperator(2, shapein=2),
                HomothetyOperator(3, shapein=3),
            ),
            None,
        ),
        (
            (I, HomothetyOperator(2, shapein=2), HomothetyOperator(3, shapein=3)),
            (1, 2, 3),
        ),
        (
            (HomothetyOperator(1, shapein=1), 2 * I, HomothetyOperator(3, shapein=3)),
            (1, 2, 3),
        ),
        (
            (HomothetyOperator(1, shapein=1), HomothetyOperator(2, shapein=2), 3 * I),
            (1, 2, 3),
        ),
    ],
)
def test_partition1(ref_partition1, ops, partition):
    op = BlockDiagonalOperator(ops, partitionin=partition, axisin=0)
    assert_equal(op.todense(6), ref_partition1, str(op))


@pytest.mark.parametrize(
    'axisp, partition',
    [
        (0, (1, 1, 1)),
        (1, (1, 2, 1)),
        (2, (2, 2, 1)),
        (3, (2, 3, 1)),
        (-1, (2, 3, 1)),
        (-2, (2, 2, 1)),
        (-3, (1, 2, 1)),
    ],
)
@pytest.mark.parametrize('axiss', range(4))
def test_partition2(axisp, partition, axiss):
    # in some cases in this test, partitionout cannot be inferred from
    # partitionin, because the former depends on the input rank
    input = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    op = BlockDiagonalOperator(
        3 * [Stretch(axiss)], partitionin=partition, axisin=axisp
    )
    assert_equal(op(input), Stretch(axiss)(input))


def test_partition3():
    # test axisin != axisout...
    pass


def test_partition4():
    o1 = HomothetyOperator(1, shapein=1)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=3)

    @flags.separable
    class Op(Operator):
        pass

    op = Op()
    p = BlockDiagonalOperator([o1, o2, o3], axisin=0)
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)


@pytest.mark.parametrize(
    'axis, shape',
    [
        (-3, (3, 2, 2)),
        (-2, (2, 3, 2)),
        (-1, (2, 2, 3)),
        (0, (3, 2, 2)),
        (1, (2, 3, 2)),
        (2, (2, 2, 3)),
    ],
)
def test_block1(axis, shape):
    ops = [HomothetyOperator(i, shapein=(2, 2)) for i in range(1, 4)]
    op = BlockDiagonalOperator(ops, new_axisin=axis)
    assert_equal(op.shapein, shape)
    assert_equal(op.shapeout, shape)


@pytest.mark.parametrize('axisp', [0, 1, 2, 3, -1, -2, -3])
@pytest.mark.parametrize('axiss', [0, 1, 2])
def test_block2(axisp, axiss):
    shape = (3, 4, 5, 6)
    i = np.arange(np.product(shape)).reshape(shape)
    op = BlockDiagonalOperator(shape[axisp] * [Stretch(axiss)], new_axisin=axisp)
    axisp_ = axisp if axisp >= 0 else axisp + 4
    axiss_ = axiss if axisp_ > axiss else axiss + 1
    assert_equal(op(i), Stretch(axiss_)(i))


def test_block3():
    # test new_axisin != new_axisout...
    pass


def test_block4():
    o1 = HomothetyOperator(1, shapein=2)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=2)

    @flags.separable
    class Op(Operator):
        pass

    op = Op()
    p = BlockDiagonalOperator([o1, o2, o3], new_axisin=0)
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)


def test_block_column1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)

    with pytest.raises(ValueError):
        BlockColumnOperator([I2, 2 * I3], axisout=0)

    with pytest.raises(ValueError):
        BlockColumnOperator([I2, 2 * I3], new_axisout=0)


def test_block_column2():
    p = np.array([[1, 0], [0, 2], [1, 0]])
    o = asoperator(p)
    e = BlockColumnOperator([o, 2 * o], axisout=0)
    assert_equal(e.todense(), np.vstack([p, 2 * p]))
    assert_equal(e.T.todense(), e.todense().T)
    e = BlockColumnOperator([o, 2 * o], new_axisout=0)
    assert_equal(e.todense(), np.vstack([p, 2 * p]))
    assert_equal(e.T.todense(), e.todense().T)


def test_block_row1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)

    with pytest.raises(ValueError):
        BlockRowOperator([I2, 2 * I3], axisin=0)

    with pytest.raises(ValueError):
        BlockRowOperator([I2, 2 * I3], new_axisin=0)


def test_block_row2():
    p = np.array([[1, 0], [0, 2], [1, 0]])
    o = asoperator(p)
    r = BlockRowOperator([o, 2 * o], axisin=0)
    assert_equal(r.todense(), np.hstack([p, 2 * p]))
    assert_equal(r.T.todense(), r.todense().T)
    r = BlockRowOperator([o, 2 * o], new_axisin=0)
    assert_equal(r.todense(), np.hstack([p, 2 * p]))
    assert_equal(r.T.todense(), r.todense().T)


@pytest.mark.parametrize('operation', [AdditionOperator, MultiplicationOperator])
@pytest.mark.parametrize('p1', [(None, None), (2, None), (None, 3), (2, 3)])
@pytest.mark.parametrize('p2', [(None, None), (2, None), (None, 3), (2, 3)])
def test_partition_implicit_commutative(operation, p1, p2):
    ops = [I, 2 * I]

    for cls, aout, ain, pout1, pin1, pout2, pin2 in zip(
        (BlockRowOperator, BlockDiagonalOperator, BlockColumnOperator),
        (None, 0, 0),
        (0, 0, None),
        (None, p1, p1),
        (p1, p1, None),
        (None, p2, p2),
        (p2, p2, None),
    ):
        op1 = BlockOperator(
            ops,
            partitionout=pout1,
            partitionin=pin1,
            axisin=ain,
            axisout=aout,
        )
        op2 = BlockOperator(
            ops,
            partitionout=pout2,
            partitionin=pin2,
            axisin=ain,
            axisout=aout,
        )
        op = operation([op1, op2])
        assert type(op) is cls
        if op.partitionin is None:
            assert op1.partitionin is op2.partitionin is None
        else:
            assert op.partitionin == merge_none(p1, p2)
        if op.partitionout is None:
            assert op1.partitionout is op2.partitionout is None
        else:
            assert op.partitionout == merge_none(p1, p2)


@pytest.mark.parametrize('pin1', [(None, None), (2, None), (None, 3), (2, 3)])
@pytest.mark.parametrize('pout2', [(None, None), (2, None), (None, 3), (2, 3)])
def test_partition_implicit_composition(pin1, pout2):
    ops = [I, 2 * I]

    for cls1, cls2, cls, aout1, ain1, aout2, ain2, pout1, pin2, in zip(
        (
            BlockRowOperator,
            BlockRowOperator,
            BlockDiagonalOperator,
            BlockDiagonalOperator,
        ),
        (
            BlockDiagonalOperator,
            BlockColumnOperator,
            BlockDiagonalOperator,
            BlockColumnOperator,
        ),
        (
            BlockRowOperator,
            HomothetyOperator,
            BlockDiagonalOperator,
            BlockColumnOperator,
        ),
        (None, None, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, None, 0, None),
        (None, None, pin1, pin1),
        (pout2, None, pout2, None),
    ):
        op1 = BlockOperator(
            ops,
            partitionin=pin1,
            partitionout=pout1,
            axisout=aout1,
            axisin=ain1,
        )
        assert type(op1) is cls1

        op2 = BlockOperator(
            ops,
            partitionout=pout2,
            partitionin=pin2,
            axisout=aout2,
            axisin=ain2,
        )
        assert type(op2) is cls2

        op = op1 * op2
        assert isinstance(op, cls)

        if not isinstance(op, BlockOperator):
            return
        pout = None if isinstance(op, BlockRowOperator) else merge_none(pin1, pout2)
        pin = None if isinstance(op, BlockColumnOperator) else merge_none(pin1, pout2)
        assert pout == op.partitionout
        assert pin == op.partitionin


@pytest.mark.parametrize(
    'op',
    [Operator(shapein=10, flags='square'), Operator(flags='linear,square', shapein=10)],
)
@pytest.mark.parametrize(
    'cls1, cls2, cls3',
    [
        (BlockRowOperator, BlockDiagonalOperator, BlockRowOperator),
        3 * (BlockDiagonalOperator,),
        (BlockDiagonalOperator, BlockColumnOperator, BlockColumnOperator),
        (BlockRowOperator, BlockColumnOperator, AdditionOperator),
    ],
)
def test_mul(op, cls1, cls2, cls3):
    operation = CompositionOperator if op.flags.linear else MultiplicationOperator
    op1 = cls1(3 * [op], axisin=0)
    op2 = cls2(3 * [op], axisout=0)
    result = op1 * op2
    assert type(result) is cls3
    assert type(result.operands[0]) is operation
