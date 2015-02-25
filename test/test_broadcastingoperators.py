from __future__ import division

import itertools
import numpy as np
import operator
from numpy.testing import assert_equal, assert_raises
from pyoperators import (
    AdditionOperator, BlockDiagonalOperator, CompositionOperator,
    ConstantOperator, DiagonalOperator, DiagonalNumexprOperator,
    HomothetyOperator, IdentityOperator, MaskOperator, MultiplicationOperator,
    Operator, PackOperator, UnpackOperator, ZeroOperator, I, O)
from pyoperators.core import BroadcastingBase
from pyoperators.flags import linear, square
from pyoperators.rules import rule_manager
from pyoperators.utils import float_or_complex_dtype, product
from pyoperators.utils.testing import (
    assert_eq, assert_is, assert_is_instance, assert_is_none, assert_is_not,
    assert_is_type, assert_not_in, assert_same)
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
        assert_is_type(op, e)
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

    x = np.arange(120.).reshape(2, 3, 4, 5) / 2

    def func(cls, d1, d2):
        op = {AdditionOperator: operator.add,
              CompositionOperator: operator.mul,
              MultiplicationOperator: operator.mul}[cls]
        d = cls([d1, d2])
        if type(d1) is DiagonalOperator:
            assert_is_type(d, DiagonalOperator)
        elif type(d1) is HomothetyOperator:
            assert_is_type(d, HomothetyOperator)
        elif op is CompositionOperator:
            assert_is_type(d, IdentityOperator)
        else:
            assert_is_type(d, HomothetyOperator)

        data = op(d1.data.T, d2.data.T).T \
               if 'rightward' in (d1.broadcast, d2.broadcast) \
               else op(d1.data, d2.data)
        assert_same(d.data, data)
        if cls is CompositionOperator:
            assert_same(d(x), d1(d2(x)))
        else:
            assert_same(d(x), op(d1(x), d2(x)))
    for op in (AdditionOperator, CompositionOperator):#, MultiplicationOperator):
        for d1, d2 in itertools.combinations(ops, 2):
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


def test_homothety_rules1():
    models = 1. * I + I, -I, (-2) * I, -(2 * I), 1. * I - I, 1. * I - 2 * I
    results = [6, -3, -6, -6, 0, -3]

    def func(model, result, i):
        o = model(i)
        assert_eq(o, result, str((model, i)))
        assert_eq(o.dtype, int, str((model, i)))
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield func, model, result, i


def test_homothety_rules2():
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


def test_homothety_rules3():
    @linear
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


def test_constant_rules1():
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


def test_constant_rules2():
    H = HomothetyOperator
    C = CompositionOperator
    D = DiagonalOperator
    cs = (ConstantOperator(3),
          ConstantOperator([1, 2, 3], broadcast='leftward'),
          ConstantOperator(np.ones((2, 3))))
    os = (I, H(2, shapein=(2, 3)) * Operator(direct=np.square, shapein=(2, 3),
                                             flags='linear,square'), H(5))
    results = (((H, 3), (C, (H, 6)), (H, 15)),
               ((D, [1, 2, 3]), (C, (D, [2, 4, 6])), (D, [5, 10, 15])),
               ((IdentityOperator, 1), (C, (H, 2)), (H, 5)))
    v = np.arange(6).reshape((2, 3))

    def func(c, o, r):
        op = MultiplicationOperator([c, o])
        assert_eq(op(v), c.data*o(v))
        assert_is_type(op, r[0])
        if type(op) is CompositionOperator:
            op = op.operands[0]
            r = r[1]
            assert_is_type(op, r[0])
        assert_eq, op.data, r[1]
    for c, rs in zip(cs, results):
        for o, r in zip(os, rs):
            yield func, c, o, r


def _test_constant_rules3():
    @square
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


def test_packing():
    valids = np.array([[False, True, True], [False, True, True]])
    valids = valids.ravel(), valids
    xs = np.array([[1, 2, 3], [4, 5, 6]])
    xs = xs.ravel(), xs
    shapes = (), (4,), (4, 5)
    broadcasts = 'disabled', 'leftward', 'rightward'
    expected = np.array([2, 3, 5, 6])

    def func(valid, x, shape, broadcast):
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
            assert_raises(ValueError, p, x_)
            return
        assert_equal(p(x_), expected_)

        assert_is_type(p.T, UnpackOperator)
        assert_equal(p.T.broadcast, p.broadcast)
        assert_equal(p.T(expected_), masking(x_))

        u = UnpackOperator(valid, broadcast=broadcast)
        assert_is_type(u.T, PackOperator)
        assert_equal(u.T.broadcast, u.broadcast)
        assert_equal(u(expected_), masking(x_))
        assert_equal(u.T(x_), expected_)

    for valid, x in zip(valids, xs):
        for shape in shapes:
            for broadcast in broadcasts:
                yield func, valid, x, shape, broadcast


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
            assert_equal(op.dtype, float_or_complex_dtype(t))
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
        with rule_manager(none=True):
            q = operation([a, b])
        assert_equal(p.todense(), q.todense())

    for cls, (commutative, left, right) in zip(clss, valids):
        for ndims in range(3):
            shape = tuple(range(2, 2 + ndims))

            def sfunc1(ndim):
                s = list(range(2, ndim + 2))
                data = np.arange(product(s)).reshape(s) + 2
                if cls is MaskOperator:
                    data = (data % 2).astype(bool)
                return data

            def sfunc2(ndim):
                s = list(range(2 + ndims - ndim, 2 + ndims))
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
