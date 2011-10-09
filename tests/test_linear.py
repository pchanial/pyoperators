from __future__ import division

import nose
import numpy as np

from nose.plugins.skip import SkipTest
from numpy.testing import assert_, assert_array_equal
from pyoperators import (
    Operator,
    IdentityOperator,
    ZeroOperator,
    DiagonalOperator,
    MaskOperator,
    PackOperator,
    UnpackOperator,
)
from pyoperators.decorators import linear


def test_masking():

    mask = MaskOperator(0)
    yield assert_, isinstance(mask, IdentityOperator)
    mask = MaskOperator(0, shapein=(32, 32), dtype=np.float32)
    yield assert_, isinstance(mask, IdentityOperator)
    yield assert_, mask.shapein == (32, 32)
    yield assert_, mask.dtype == np.float32

    mask = MaskOperator(1)
    yield assert_, isinstance(mask, ZeroOperator)
    mask = MaskOperator(1, shapein=(32, 32), dtype=np.float32)
    yield assert_, isinstance(mask, ZeroOperator)
    yield assert_, mask.shapein == (32, 32)
    yield assert_, mask.dtype == np.float32

    b = np.array([3.0, 4.0, 1.0, 0.0, 3.0, 2.0])
    c = np.array([3.0, 4.0, 0.0, 0.0, 3.0, 0.0])
    mask = MaskOperator(np.array([0, 0.0, 1.0, 1.0, 0.0, 1], dtype=np.int8))
    yield assert_, np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([1, 1.0, 0.0, 0.0, 1.0, 0]))
    yield assert_, np.all(mask(b) == c)
    mask = MaskOperator(np.array([False, False, True, True, False, True]))
    yield assert_, np.all(mask(b) == c)

    b = np.array([[3.0, 4.0], [1.0, 0.0], [3.0, 2.0]])
    c = np.array([[3.0, 4.0], [0.0, 0.0], [3.0, 0.0]])
    mask = MaskOperator(np.array([[0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype='int8'))
    yield assert_, np.all(mask(b) == c)
    mask = DiagonalOperator(np.array([[1, 1.0], [0.0, 0.0], [1.0, 0.0]]))
    yield assert_, np.all(mask(b) == c)
    mask = MaskOperator(np.array([[False, False], [True, True], [False, True]]))
    yield assert_, np.all(mask(b) == c)

    b = np.array([[[3, 4.0], [1.0, 0.0]], [[3.0, 2], [-1, 9]]])
    c = np.array([[[3, 4.0], [0.0, 0.0]], [[3.0, 0], [0, 0]]])
    mask = MaskOperator(np.array([[[0, 0.0], [1.0, 1.0]], [[0.0, 1], [1, 1]]], int))
    yield assert_, np.all(mask(b) == c)

    mask = DiagonalOperator(np.array([[[1, 1], [0.0, 0]], [[1, 0], [0, 0]]]))
    yield assert_, np.all(mask(b) == c)
    mask = MaskOperator(
        np.array([[[False, False], [True, True]], [[False, True], [True, True]]])
    )
    yield assert_, np.all(mask(b) == c)

    c = mask(b, b)
    yield assert_, id(b) == id(c)


def test_masking2():
    m = MaskOperator([True, False, True])
    yield assert_, m * m is m


def test_diagonal1():
    d = DiagonalOperator([1.0, 2.0, 3.0])
    yield assert_, isinstance(2 * d, DiagonalOperator)
    yield assert_, d * d is not d
    yield assert_, isinstance(d * d, DiagonalOperator)
    yield assert_, isinstance(3 + 2 * d * 3.0 * d + 2 * d + 2, DiagonalOperator)


def test_zero1():
    z = ZeroOperator()
    o = Operator()
    yield assert_, isinstance(z * o, ZeroOperator)
    yield assert_, not isinstance(o * z, ZeroOperator)


def test_zero2():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6)
    zo = z * o
    yield assert_, isinstance(zo, ZeroOperator)
    yield assert_, zo.shapein == o.shapein and zo.shapeout == o.shapeout


def test_zero3():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator()
    zo = z * o
    yield assert_, isinstance(zo, ZeroOperator)
    yield assert_, zo.shapein == z.shapein and zo.shapeout == z.shapeout


def test_zero4():
    z = ZeroOperator()
    o = Operator(flags={'LINEAR': True})
    yield assert_, isinstance(z * o, ZeroOperator)
    yield assert_, isinstance(o * z, ZeroOperator)


def test_zero5():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6, flags={'LINEAR': True})
    zo = z * o
    oz = o * z
    yield assert_, isinstance(zo, ZeroOperator)
    yield assert_, zo.shapein == o.shapein and zo.shapeout == o.shapeout
    yield assert_, isinstance(oz, ZeroOperator)
    yield assert_, oz.shapein == o.shapein and oz.shapeout == o.shapeout


def test_zero6():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator(flags={'LINEAR': True})
    zo = z * o
    oz = o * z
    yield assert_, isinstance(zo, ZeroOperator)
    yield assert_, zo.shapein == z.shapein and zo.shapeout == z.shapeout
    yield assert_, isinstance(oz, ZeroOperator)
    yield assert_, oz.shapein == z.shapein and oz.shapeout == z.shapeout


def test_zero7():
    z = ZeroOperator()

    @linear
    class Op(Operator):
        def direct(self, input, output):
            output[:] = np.concatenate([input, 2 * input])

        def transpose(self, input, output):
            output[:] = input[0 : output.size]

        def reshapein(self, shapein):
            if shapein is None:
                return None
            s = list(shapein)
            s[0] *= 2
            return s

        def reshapeout(self, shapeout):
            if shapeout is None:
                return None
            s = list(shapeout)
            s[0] //= 2
            return s

    o = Op()
    zo = z * o
    oz = o * z
    v = np.ones(4)
    yield assert_array_equal, zo(v), z(o(v))
    yield assert_array_equal, oz(v), o(z(v))


def test_zero7b():
    z = ZeroOperator()

    @linear
    class Op(Operator):
        def direct(self, input, output):
            output[:] = np.concatenate([input, 2 * input])

        def transpose(self, input, output):
            output[:] = input[0 : output.size]

        def reshapein(self, shapein):
            if shapein is None:
                return None
            s = list(shapein)
            s[0] *= 2
            return s

        def reshapeout(self, shapeout):
            if shapeout is None:
                return None
            s = list(shapeout)
            s[0] //= 2
            return s

    o = Op()
    zo = z * o
    oz = o * z
    v = np.ones(4)
    yield assert_array_equal, zo.T(v), o.T(z.T(v))
    yield assert_array_equal, oz.T(v), z.T(o.T(v))


def test_zero8():
    z = ZeroOperator()
    yield assert_, z * z is z


def test_packing():

    p = PackOperator([False, True, True, False, True])
    yield assert_, p.T.__class__ == UnpackOperator
    yield assert_, np.allclose(p([1, 2, 3, 4, 5]), [1, 4])
    yield assert_, np.allclose(p.T([1, 4]), [1, 0, 0, 4, 0])

    u = UnpackOperator([False, True, True, False, True])
    yield assert_, u.T.__class__ == PackOperator
    yield assert_, np.allclose(u([1, 4]), [1, 0, 0, 4, 0])
    yield assert_, np.allclose(u.T([1, 2, 3, 4, 5]), [1, 4])


if __name__ == "__main__":
    nose.run(argv=['', __file__])
