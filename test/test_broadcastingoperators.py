from __future__ import division

import itertools
import numpy as np
from numpy.testing import assert_equal
from pyoperators import (
    AdditionOperator, BlockDiagonalOperator, CompositionOperator,
    ConstantOperator, DiagonalOperator, DiagonalNumexprOperator, GroupOperator,
    HomothetyOperator, IdentityOperator, MaskOperator, MultiplicationOperator,
    PackOperator, UnpackOperator, ZeroOperator)
from pyoperators.core import BroadcastingBase
from pyoperators.utils import float_dtype, product
from pyoperators.utils.testing import assert_is_instance, assert_is_none
from .common import HomothetyOutplaceOperator

clss = (ConstantOperator, DiagonalOperator, DiagonalNumexprOperator,
        HomothetyOperator, IdentityOperator, MaskOperator, ZeroOperator)
sameshapes = ((False, True), (True, True), (True, True), (True, True))
types = (bool, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
         np.float64, np.float128, np.complex128, np.complex256)


def get_operator(cls, data, **keywords):
    if cls is DiagonalNumexprOperator:
        args = (data, '3*data')
    elif cls is HomothetyOperator:
        args = (data.flat[0],)
    elif cls in (IdentityOperator, ZeroOperator):
        args = ()
    else:
        args = (data,)
    return cls(*args, **keywords)


def test_dtype():
    x = np.array([3, 0, 2])

    def func(c, t):
        op = get_operator(c, x.astype(t))
        c_ = type(op)
        if c_ in (IdentityOperator, ZeroOperator):
            expected_dtype = int
        elif c_ is MaskOperator:
            expected_dtype = bool
        else:
            expected_dtype = t
        assert_equal(op.data.dtype, expected_dtype)

        if c_ in (IdentityOperator, MaskOperator, ZeroOperator):
            assert_equal(op.dtype, None)
        elif c_ is DiagonalNumexprOperator:
            assert_equal(op.dtype, float_dtype(t))
        else:
            assert_equal(op.dtype, t)

    for c in clss:
        for t in types:
            yield func, c, t


def test_shape():
    shapes = (), (3,), (3, 2)
    broadcasts = None, 'leftward', 'rightward'

    def func(c, s, b, sameshapein, sameshapeout):
        x = np.arange(product(s)).reshape(s)
        op = get_operator(c, x, broadcast=b)
        if len(s) == 0 or c in (HomothetyOperator, IdentityOperator,
                                ZeroOperator):
            assert_equal(op.broadcast, 'scalar')
            assert_is_none(op.shapein)
            assert_is_none(op.shapeout)
        elif b in ('leftward', 'rightward'):
            assert_equal(op.broadcast, b)
            assert_is_none(op.shapein)
            assert_is_none(op.shapeout)
        else:
            assert_equal(op.broadcast, 'disabled')
            if sameshapein:
                assert_equal(op.shapein, s)
            if sameshapeout:
                assert_equal(op.shapeout, s)
    for c, (sameshapein, sameshapeout) in zip(clss, sameshapes):
        for s in shapes:
            for b in broadcasts:
                yield func, c, s, b, sameshapein, sameshapeout


def test_partition():
    clss = (ConstantOperator, DiagonalOperator, DiagonalNumexprOperator,
            HomothetyOperator, IdentityOperator, MaskOperator, PackOperator,
            UnpackOperator)
    valids = ((True, False, False), (True, True, True), (True, True, True),
              (True, True, True), (True, True, True), (True, True, True),
              (True, False, True), (True, True, False))

    def func(a, b, operation, apply_rule):
        p = operation([a, b])
        if not apply_rule:
            if isinstance(a, IdentityOperator) or \
               isinstance(b, IdentityOperator):
                return
            assert not isinstance(p, BlockDiagonalOperator)
            return
        assert_is_instance(p, BlockDiagonalOperator)
        q = operation([a, GroupOperator(b)])
        assert_equal(p.todense(), q.todense())

    for cls, (commutative, left, right) in zip(clss, valids):
        for ndims in range(3):
            shape = tuple(range(2, 2 + ndims))

            def sfunc1(ndim):
                s = range(2, ndim + 2)
                data = np.arange(product(s)).reshape(s) + 2
                if cls is MaskOperator:
                    data = (data % 2).astype(bool)
                return data

            def sfunc2(ndim):
                s = range(2 + ndims - ndim, 2 + ndims)
                data = np.arange(product(s)).reshape(s) + 2
                if cls is MaskOperator:
                    data = (data % 2).astype(bool)
                return data

            if cls in (HomothetyOperator, IdentityOperator):
                ops = [get_operator(cls, np.array(2))]
            else:
                ops = [get_operator(cls, sfunc1(ndim))
                       for ndim in range(ndims+1)] + \
                      [get_operator(cls, sfunc2(ndim), broadcast='leftward')
                       for ndim in range(1, ndims+1)] + \
                      [get_operator(cls, sfunc1(ndim), broadcast='rightward')
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

            for o, b in itertools.product(ops, block):
                if (o.shapein is None or o.shapein == b.shapein) and \
                   (o.shapeout is None or o.shapeout == b.shapeout):
                    yield func, o, b, AdditionOperator, commutative
                    yield func, o, b, MultiplicationOperator, commutative
                if o.shapein is None or o.shapein == b.shapeout:
                    yield func, o, b, CompositionOperator, right
                if o.shapeout is None or b.shapein == o.shapeout:
                    yield func, b, o, CompositionOperator, left


def test_as_strided():
    shapes = {'leftward': (2, 4, 3, 4, 2, 2),
              'rightward': (3, 2, 2, 3, 1, 2)}

    def func(b):
        o = BroadcastingBase(np.arange(6).reshape((3, 1, 2, 1)), b)
        s = shapes[b]
        if b == 'leftward':
            v = o.data*np.ones(s)
        else:
            v = (o.data.T * np.ones(s, int).T).T
        assert_equal(o._as_strided(s), v)
    for b in ('rightward', 'leftward'):
        yield func, b
