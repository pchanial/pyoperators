import itertools
import numpy as np

from .common import HomothetyOutplaceOperator, Stretch
from pyoperators import (
    decorators, Operator, AdditionOperator, BlockColumnOperator,
    BlockDiagonalOperator, BlockRowOperator, CompositionOperator,
    ConstantOperator, DiagonalOperator, HomothetyOperator, IdentityOperator,
    MultiplicationOperator, I, asoperator)
from pyoperators.core import BlockOperator
from pyoperators.utils import merge_none
from pyoperators.utils.testing import (
    assert_eq, assert_is_instance, assert_raises)


def test_partition1():
    o1 = HomothetyOperator(1, shapein=1)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=3)
    r = DiagonalOperator([1, 2, 2, 3, 3, 3]).todense()

    def func(ops, p):
        op = BlockDiagonalOperator(ops, partitionin=p, axisin=0)
        assert_eq(op.todense(6), r, str(op))
    for ops, p in zip(
            ((o1, o2, o3), (I, o2, o3), (o1, 2*I, o3), (o1, o2, 3*I)),
            (None, (1, 2, 3), (1, 2, 3), (1, 2, 3))):
        yield func, ops, p


def test_partition2():
    # in some cases in this test, partitionout cannot be inferred from
    # partitionin, because the former depends on the input rank
    i = np.arange(3*4*5*6).reshape(3, 4, 5, 6)

    def func(axisp, p, axiss):
        op = BlockDiagonalOperator(3*[Stretch(axiss)], partitionin=p,
                                   axisin=axisp)
        assert_eq(op(i), Stretch(axiss)(i))
    for axisp, p in zip(
            (0, 1, 2, 3, -1, -2, -3),
            ((1, 1, 1), (1, 2, 1), (2, 2, 1), (2, 3, 1), (2, 3, 1), (2, 2, 1),
             (1, 2, 1), (1, 1, 1))):
        for axiss in (0, 1, 2, 3):
            yield func, axisp, p, axiss


def test_partition3():
    # test axisin != axisout...
    pass


def test_partition4():
    o1 = HomothetyOperator(1, shapein=1)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=3)

    @decorators.separable
    class Op(Operator):
        pass
    op = Op()
    p = BlockDiagonalOperator([o1, o2, o3], axisin=0)
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)


def test_block1():
    ops = [HomothetyOperator(i, shapein=(2, 2)) for i in range(1, 4)]

    def func(axis, s):
        op = BlockDiagonalOperator(ops, new_axisin=axis)
        assert_eq(op.shapein, s)
        assert_eq(op.shapeout, s)
    for axis, s in zip(
            range(-3, 3),
            ((3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 2, 2), (2, 3, 2),
             (2, 2, 3))):
        yield func, axis, s


def test_block2():
    shape = (3, 4, 5, 6)
    i = np.arange(np.product(shape)).reshape(shape)

    def func(axisp, axiss):
        op = BlockDiagonalOperator(shape[axisp]*[Stretch(axiss)],
                                   new_axisin=axisp)
        axisp_ = axisp if axisp >= 0 else axisp + 4
        axiss_ = axiss if axisp_ > axiss else axiss + 1
        assert_eq(op(i), Stretch(axiss_)(i))
    for axisp in (0, 1, 2, 3, -1, -2, -3):
        for axiss in (0, 1, 2):
            yield func, axisp, axiss


def test_block3():
    # test new_axisin != new_axisout...
    pass


def test_block4():
    o1 = HomothetyOperator(1, shapein=2)
    o2 = HomothetyOperator(2, shapein=2)
    o3 = HomothetyOperator(3, shapein=2)

    @decorators.separable
    class Op(Operator):
        pass
    op = Op()
    p = BlockDiagonalOperator([o1, o2, o3], new_axisin=0)
    r = (op + p + op) * p
    assert isinstance(r, BlockDiagonalOperator)


def test_block_column1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, BlockColumnOperator, [I2, 2*I3], axisout=0)
    assert_raises(ValueError, BlockColumnOperator, [I2, 2*I3], new_axisout=0)


def test_block_column2():
    p = np.matrix([[1, 0], [0, 2], [1, 0]])
    o = asoperator(np.matrix(p))
    e = BlockColumnOperator([o, 2*o], axisout=0)
    assert_eq(e.todense(), np.vstack([p, 2*p]))
    assert_eq(e.T.todense(), e.todense().T)
    e = BlockColumnOperator([o, 2*o], new_axisout=0)
    assert_eq(e.todense(), np.vstack([p, 2*p]))
    assert_eq(e.T.todense(), e.todense().T)


def test_block_row1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, BlockRowOperator, [I2, 2*I3], axisin=0)
    assert_raises(ValueError, BlockRowOperator, [I2, 2*I3], new_axisin=0)


def test_block_row2():
    p = np.matrix([[1, 0], [0, 2], [1, 0]])
    o = asoperator(np.matrix(p))
    r = BlockRowOperator([o, 2*o], axisin=0)
    assert_eq(r.todense(), np.hstack([p, 2*p]))
    assert_eq(r.T.todense(), r.todense().T)
    r = BlockRowOperator([o, 2*o], new_axisin=0)
    assert_eq(r.todense(), np.hstack([p, 2*p]))
    assert_eq(r.T.todense(), r.todense().T)


def test_partition_implicit_commutative():
    partitions = (None, None), (2, None), (None, 3), (2, 3)
    ops = [I, 2*I]

    def func(op1, op2, p1, p2, cls):
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
    for operation in (AdditionOperator, MultiplicationOperator):
        for p1 in partitions:
            for p2 in partitions:
                for cls, aout, ain, pout1, pin1, pout2, pin2 in zip(
                        (BlockRowOperator, BlockDiagonalOperator,
                         BlockColumnOperator),
                        (None, 0, 0), (0, 0, None), (None, p1, p1),
                        (p1, p1, None), (None, p2, p2), (p2, p2, None)):
                    op1 = BlockOperator(
                        ops, partitionout=pout1, partitionin=pin1, axisin=ain,
                        axisout=aout)
                    op2 = BlockOperator(
                        ops, partitionout=pout2, partitionin=pin2, axisin=ain,
                        axisout=aout)
                    yield func, op1, op2, p1, p2, cls


def test_partition_implicit_composition():
    partitions = (None, None), (2, None), (None, 3), (2, 3)
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
                     BlockDiagonalOperator),
                    (BlockDiagonalOperator, BlockColumnOperator,
                     BlockDiagonalOperator, BlockColumnOperator),
                    (BlockRowOperator, HomothetyOperator,
                     BlockDiagonalOperator, BlockColumnOperator),
                    (None, None, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
                    (0, None, 0, None), (None, None, pin1, pin1),
                    (pout2, None, pout2, None)):
                op1 = BlockOperator(ops, partitionin=pin1, partitionout=pout1,
                                    axisout=aout1, axisin=ain1)
                op2 = BlockOperator(ops, partitionout=pout2, partitionin=pin2,
                                    axisout=aout2, axisin=ain2)
                yield func, op1, op2, pin1, pout2, cls


def test_partition_diagonaloperator_broadcast():
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
        shape = tuple(range(2, 2+ndims))
        sfunc1 = lambda ndim: np.arange(np.product(range(2, ndim+2))).reshape(
            range(2, ndim+2)) + 2
        sfunc2 = lambda ndim: np.arange(np.product(range(
            2+ndims-ndim, 2+ndims))).reshape(range(2+ndims-ndim, 2+ndims)) + 2
        diag = [DiagonalOperator(sfunc1(ndim)) for ndim in range(ndims+1)] + \
               [DiagonalOperator(sfunc2(ndim), broadcast='leftward')
                for ndim in range(1, ndims+1)] + \
               [DiagonalOperator(sfunc1(ndim), broadcast='rightward')
                for ndim in range(1, ndims+1)]

        def toone(index):
            list_ = list(shape)
            list_[index] = 1
            return list_

        def remove(index):
            list_ = list(shape)
            list_.pop(index)
            return list_
        block = \
            [BlockDiagonalOperator([HomothetyOutplaceOperator(
                v, shapein=toone(axis)) for v in range(2, 2+shape[axis])],
                axisin=axis, partitionin=shape[axis]*[1])
             for axis in range(-ndims, ndims)] + \
            [BlockDiagonalOperator([HomothetyOutplaceOperator(
                v, shapein=remove(axis)) for v in range(2, 2+shape[axis])],
                new_axisin=axis, partitionin=shape[axis]*[1])
             for axis in range(-ndims, ndims)]

        for d, b in itertools.product(diag, block):
            if d.broadcast == 'disabled' and d.shapein != b.shapein:
                continue
            yield func1, d, b
            yield func2, d, b
            yield func3, b, d


def test_partition_constantoperator_broadcast():
    def func1(c, b):
        p = c * b
        if c.flags.shape_output != 'explicit' or c.shapeout != b.shapeout:
            assert_is_instance(p, ConstantOperator)
            assert_eq(p.data, c.data)
            assert_eq(p.shapein, b.shapein)
            return
        assert_is_instance(p, BlockDiagonalOperator)
        c_ = c.copy(); c_.rules = c_.rules.copy()
        b_ = b.copy(); b_.rules = b_.rules.copy()
        del c_.rules[CompositionOperator]
        del b_.rules[CompositionOperator]
        p_ = c_ * b_
        assert_eq(p.todense(), p_.todense())

    def func2(c, b):
        p = c + b
        assert_is_instance(p, BlockDiagonalOperator)
        c_ = c.copy(); c_.rules = c_.rules.copy()
        b_ = b.copy(); b_.rules = b_.rules.copy()
        del c_.rules[AdditionOperator]
        del b_.rules[AdditionOperator]
        p_ = c_ + b_
        assert_eq(p.todense(), p_.todense())

    for ndims in range(4):
        shape = tuple(range(2, 2+ndims))
        sfunc1 = lambda ndim: np.arange(np.product(range(
            2, ndim+2))).reshape(range(2, ndim+2)) + 2
        sfunc2 = lambda ndim: np.arange(np.product(range(
            2+ndims-ndim, 2+ndims))).reshape(range(2+ndims-ndim, 2+ndims)) + 2
        const = [ConstantOperator(sfunc1(ndim)) for ndim in range(ndims+1)] + \
                [ConstantOperator(sfunc2(ndim), broadcast='leftward')
                 for ndim in range(1, ndims+1)] + \
                [ConstantOperator(sfunc1(ndim), broadcast='rightward')
                 for ndim in range(1, ndims+1)]

        def toone(index):
            list_ = list(shape)
            list_[index] = 1
            return list_

        def remove(index):
            list_ = list(shape)
            list_.pop(index)
            return list_
        block = \
            [BlockDiagonalOperator([HomothetyOutplaceOperator(
                v, shapein=toone(axis)) for v in range(2, 2+shape[axis])],
                axisin=axis, partitionin=shape[axis]*[1])
             for axis in range(-ndims, ndims)] + \
            [BlockDiagonalOperator([HomothetyOutplaceOperator(
                v, shapein=remove(axis)) for v in range(2, 2+shape[axis])],
                new_axisin=axis, partitionin=shape[axis]*[1])
             for axis in range(-ndims, ndims)]

        for c, b in itertools.product(const, block):
            yield func1, c, b
            if c.broadcast == 'disabled' and c.shapeout != b.shapeout:
                continue
            yield func2, c, b
