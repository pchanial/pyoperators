import numpy as np
from numpy.testing import assert_equal

from pyoperators import (
    HardThresholdingOperator,
    NumexprOperator,
    RoundOperator,
    SoftThresholdingOperator,
)


def test_rounding():
    a = np.array([-3.5, -3, -2.6, -2.5, -2.4, 0, 0.2, 0.5, 0.9, 1, 1.5])
    r = RoundOperator('rtz')
    yield assert_equal, r(a), [-3, -3, -2, -2, -2, 0, 0, 0, 0, 1, 1]
    # r = RoundOperator('rti')
    # yield assert_equal, r(a), [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2]
    r = RoundOperator('rtmi')
    yield assert_equal, r(a), [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1]
    r = RoundOperator('rtpi')
    yield assert_equal, r(a), [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2]
    # r = RoundOperator('rhtz')
    # yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    # r = RoundOperator('rhti')
    # yield assert_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2]
    r = RoundOperator('rhtmi')
    yield assert_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1]
    r = RoundOperator('rhtpi')
    yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2]
    r = RoundOperator('rhte')
    yield assert_equal, r(a), [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2]
    # r = RoundOperator('rhto')
    # yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    # r = RoundOperator('rhs')
    # mask = np.array([True,True,False,True,True,True,False,True,True], np.bool)
    # result = r(a)
    # yield assert_equal, result[mask], [-3,-3,-2,0,0,1,1]
    # yield assert_, result[2] in (-3,-2)
    # yield assert_, result[-4] in (0,1)


def test_numexpr1():
    d = 7.0
    op = NumexprOperator('2.*exp(input)+d', {'d': d})
    assert op(3.0) == 2 * np.exp(3.0) + d


def test_numexpr2():
    op = NumexprOperator('3*input') + NumexprOperator('2*input')
    assert_equal(op(np.arange(10)), 5 * np.arange(10))


def test_hard_thresholding():
    x = [-1.0, -0.2, -0.1, 0, 0.2, 0.1, 2, 3]
    lbda = 0.2
    H = HardThresholdingOperator(lbda)
    expected = [-1, 0, 0, 0, 0, 0, 2, 3]
    assert_equal(H(x), expected)
    x = np.array(x)
    H(x, x)
    assert_equal(x, expected)
    lbda2 = [0.3, 0.1, 2]
    shape = np.asarray(lbda2).shape
    G = HardThresholdingOperator(lbda2)
    assert_equal(G.shapein, shape)
    K = G * H
    assert_equal(K.a, np.maximum(lbda, lbda2))
    assert_equal(K.shapein, shape)
    K = H * G
    assert_equal(K.a, np.maximum(lbda, lbda2))
    assert_equal(K.shapein, shape)


def test_soft_thresholding():
    x = [-1.0, -0.2, -0.1, 0, 0.1, 0.2, 2, 3]
    lbda = np.array(0.2)
    S = SoftThresholdingOperator(lbda)
    expected = [-1, 0, 0, 0, 0, 0, 2, 3] - lbda * [-1, 0, 0, 0, 0, 0, 1, 1]
    assert_equal(S(x), expected)
    x = np.array(x)
    S(x, x)
    assert_equal(x, expected)
    lbda2 = [0.3, 0.1, 2]
    shape = np.asarray(lbda2).shape
    T = SoftThresholdingOperator(lbda2)
    assert_equal(T.shapein, shape)
