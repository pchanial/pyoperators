from __future__ import division
import numpy as np
from numpy.testing import assert_equal
from pyoperators import (
    CompositionOperator, ConstantOperator, Operator, ZeroOperator, flags, O)
from pyoperators.utils import ndarraywrap
from pyoperators.utils.testing import (
    assert_is_instance, assert_is_none, skiptest)
from .common import OPS, ndarray2, attr2

op = Operator()
ops = [OP() for OP in OPS]
zeros_left = (ZeroOperator(classout=ndarray2, attrout=attr2),
              ZeroOperator(shapein=4, classout=ndarray2, attrout=attr2))
zeros_right = (ZeroOperator(classout=ndarray2, attrout=attr2),
               ZeroOperator(classout=ndarray2, attrout=attr2, flags='square'),
               ZeroOperator(shapein=3, classout=ndarray2, attrout=attr2))


def test_zero1():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6)
    zo = z(o)
    assert_is_instance(zo, ZeroOperator)
    assert_equal(zo.shapein, o.shapein)
    assert_is_none(zo.shapeout)


def test_zero2():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator()
    zo = z(o)
    assert_is_instance(zo, ZeroOperator)
    assert_is_none(zo.shapein, 'in')
    assert_equal(zo.shapeout, z.shapeout, 'out')


def test_zero3():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator(flags='square')
    zo = z*o
    assert_is_instance(zo, ZeroOperator)
    assert_equal(zo.shapein, z.shapein, 'in')
    assert_equal(zo.shapeout, z.shapeout, 'out')


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
    assert_equal(zo.shapein, o.shapein, 'zo in')
    assert_is_none(zo.shapeout, 'zo out')
    assert_is_instance(oz, ZeroOperator, 'oz')
    assert_is_none(oz.shapein, 'oz, in')
    assert_equal(oz.shapeout, o.shapeout, 'oz, out')


def test_zero6():
    z = ZeroOperator(flags='square')

    @flags.linear
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
    assert_equal(zo.T(v), o.T(z.T(v)))
    assert_equal(oz.T(v), z.T(o.T(v)))


def test_zero7():
    z = ZeroOperator()
    assert_equal(z*z, z)


def test_zero8():
    class Op(Operator):
        pass
    o = Op()
    assert type(o + O) is Op


@skiptest
def test_merge_zero_left():
    def func(op1, op2):
        op = op1 * op2
        assert_is_instance(op, ZeroOperator)
        attr = {}
        attr.update(op2.attrout)
        attr.update(op1.attrout)
        assert_equal(op.attrout, attr)
        x = np.ones(3)
        y = ndarraywrap(4)
        op(x, y)
        y2_tmp = np.empty(4)
        y2 = np.empty(4)
        op2(x, y2_tmp)
        op1(y2_tmp, y2)
        assert_equal(y, y2)
        assert_is_instance(y, op1.classout)
    for op1 in zeros_left:
        for op2 in ops:
            yield func, op1, op2


@skiptest
def test_merge_zero_right():
    def func(op1, op2):
        op = op1 * op2
        if op1.flags.shape_output == 'unconstrained' or \
           op1.flags.shape_input != 'explicit' and \
           op2.flags.shape_output != 'explicit':
            assert_is_instance(op, CompositionOperator)
            return
        assert_is_instance(op, ConstantOperator)
        attr = {}
        attr.update(op2.attrout)
        attr.update(op1.attrout)
        assert_equal(op.attrout, attr)
        x = np.ones(3)
        y = ndarraywrap(4)
        op(x, y)
        y2_tmp = np.empty(3)
        y2 = np.empty(4)
        op2(x, y2_tmp)
        op1(y2_tmp, y2)
        assert_equal(y, y2)
        assert_is_instance(y, op1.classout)
    for op1 in ops:
        for op2 in zeros_right:
            yield func, op1, op2
