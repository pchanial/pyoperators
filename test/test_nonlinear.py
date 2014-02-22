from __future__ import division

import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
from pyoperators import (
    Cartesian2SphericalOperator, CompositionOperator, ConstantOperator,
    HardThresholdingOperator, IdentityOperator, MultiplicationOperator,
    NormalizeOperator, NumexprOperator, PowerOperator, ReciprocalOperator,
    RoundOperator, SqrtOperator, SquareOperator, SoftThresholdingOperator,
    Spherical2CartesianOperator)
from pyoperators.utils import product
from pyoperators.utils.testing import (
    assert_is_instance, assert_is_type, assert_same)


def test_cartesian_spherical():
    vecs = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (((1, 0, 0), (0, 1, 0)),))
    shapes = ((), (), (), (3,), (1, 2))

    def func(c, v, s, d):
        c2s = Cartesian2SphericalOperator(c, degrees=d)
        s2c = Spherical2CartesianOperator(c, degrees=d)
        a = s2c(c2s(v))
        assert_equal(a.shape, s + (3,))
        assert_allclose(a, v, atol=1e-16)
    for c in Cartesian2SphericalOperator.CONVENTIONS:
        for v, s in zip(vecs, shapes):
            for d in (False, True):
                yield func, c, v, s, d


def test_cartesian_spherical_error():
    assert_raises(TypeError, Cartesian2SphericalOperator, 3)
    assert_raises(ValueError, Cartesian2SphericalOperator, 'bla')
    op = Cartesian2SphericalOperator('zenith,azimuth')

    def func(i, o):
        if i.shape == (3,) and o.shape == (2,):
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_cartesian_spherical_rules():
    def func(c1, c2):
        op1 = Cartesian2SphericalOperator(c1)
        op2 = Spherical2CartesianOperator(c2)
        op = op1(op2)
        if c1 == c2:
            assert_is_type(op, IdentityOperator)
        else:
            assert_is_type(op, CompositionOperator)
    for c1 in 'zenith,azimuth', 'azimuth,elevation':
        op = Cartesian2SphericalOperator(c1)
        assert_is_type(op.I, Spherical2CartesianOperator)
        assert_equal(op.convention, c1)
        for c2 in 'zenith,azimuth', 'azimuth,elevation':
            yield func, c1, c2


def test_spherical_cartesian():
    dirs_za = ((0, 0), (20, 0), (130, 0), (10, 20), (20, 190),
               ((0, 0), (20, 0), (130, 0), (10, 20), (20, 130)),
               (((0, 0), (20, 200), (130, 300)),))
    dirs_az = ((0, 0), (0, 20), (0, 130), (20, 10), (190, 20),
               ((0, 0), (0, 20), (0, 130), (20, 10), (130, 20)),
               (((0, 0), (200, 20), (300, 130)),))
    dirs_ea = ((90, 0), (70, 0), (-40, 0), (80, 20), (70, 190),
               ((90, 0), (70, 0), (-40, 0), (80, 20), (70, 130)),
               (((90, 0), (70, 200), (-40, 300)),))
    dirs_ae = ((0, 90), (0, 70), (0, -40), (20, 80), (190, 70),
               ((0, 90), (0, 70), (0, -40), (20, 80), (130, 70)),
               (((0, 90), (200, 70), (300, -40)),))
    shapes = ((), (), (), (), (), (5,), (1, 3))

    op_ref = Spherical2CartesianOperator('zenith,azimuth')
    refs = [op_ref(np.radians(v)) for v in dirs_za]

    def func(c, v, s, d, r):
        orig = v
        if not d:
            v = np.radians(v)
        s2c = Spherical2CartesianOperator(c, degrees=d)
        c2s = Cartesian2SphericalOperator(c, degrees=d)
        assert_allclose(s2c(v), r)
        a = c2s(s2c(v))
        if not d:
            a = np.degrees(a)
        assert_equal(a.shape, s + (2,))
        assert_allclose(a, orig, atol=1e-16)
    for c, vs in (('zenith,azimuth', dirs_za),
                  ('azimuth,zenith', dirs_az),
                  ('elevation,azimuth', dirs_ea),
                  ('azimuth,elevation', dirs_ae)):
        for v, s, r in zip(vs, shapes, refs):
            for d in (False, True):
                yield func, c, v, s, d, r


def test_spherical_cartesian_error():
    assert_raises(TypeError, Spherical2CartesianOperator, 3)
    assert_raises(ValueError, Spherical2CartesianOperator, 'bla')
    op = Spherical2CartesianOperator('zenith,azimuth')

    def func(i, o):
        if i.shape == (2,) and o.shape == (3,):
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_spherical_cartesian_rules():
    def func(c1, c2):
        op1 = Spherical2CartesianOperator(c1)
        op2 = Cartesian2SphericalOperator(c2)
        op = op1(op2)
        if c1 == c2:
            assert_is_type(op, IdentityOperator)
        else:
            assert_is_type(op, CompositionOperator)
    for c1 in 'zenith,azimuth', 'azimuth,elevation':
        op = Spherical2CartesianOperator(c1)
        assert_is_type(op.I, Cartesian2SphericalOperator)
        assert_equal(op.convention, c1)
        for c2 in 'zenith,azimuth', 'azimuth,elevation':
            yield func, c1, c2


def test_rounding():
    a = np.array([-3.5, -3, -2.6, -2.5, -2.4, 0, 0.2, 0.5, 0.9, 1, 1.5])
    r = RoundOperator('rtz')
    yield assert_equal, r(a), [-3, -3, -2, -2, -2, 0, 0, 0, 0, 1, 1]
    #r = RoundOperator('rti')
    #yield assert_equal, r(a), [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2]
    r = RoundOperator('rtmi')
    yield assert_equal, r(a), [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1]
    r = RoundOperator('rtpi')
    yield assert_equal, r(a), [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2]
    #r = RoundOperator('rhtz')
    #yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    #r = RoundOperator('rhti')
    #yield assert_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2]
    r = RoundOperator('rhtmi')
    yield assert_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1]
    r = RoundOperator('rhtpi')
    yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2]
    r = RoundOperator('rhte')
    yield assert_equal, r(a), [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2]
    #r = RoundOperator('rhto')
    #yield assert_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    #r = RoundOperator('rhs')
    #mask = np.array([True,True,False,True,True,True,False,True,True], np.bool)
    #result = r(a)
    #yield assert_equal, result[mask], [-3,-3,-2,0,0,1,1]
    #yield assert_, result[2] in (-3,-2)
    #yield assert_, result[-4] in (0,1)


def test_normalize():
    n = NormalizeOperator()

    def func(shape):
        vec = np.arange(product(shape)).reshape(shape)
        exp = vec / np.sqrt(np.sum(vec ** 2, axis=-1))[..., None]
        assert_same(n(vec), exp)
    for shape in ((2,), (4,), (2, 3), (4, 5, 2)):
        yield func, shape


def test_numexpr1():
    d = 7.
    op = NumexprOperator('2.*exp(input)+d', {'d': d})
    assert op(3.) == 2*np.exp(3.)+d


def test_numexpr2():
    op = NumexprOperator('3*input') + NumexprOperator('2*input')
    assert_equal(op(np.arange(10)), 5*np.arange(10))


def test_power():
    values = -1, 0, 0.5, 1, 2, 3
    cls = (ReciprocalOperator, ConstantOperator, SqrtOperator,
           IdentityOperator, SquareOperator, PowerOperator)

    def func(n, c):
        op = PowerOperator(n)
        assert_is_type(op, c)
        if isinstance(op, PowerOperator):
            assert_equal(op.n, n)
    for v, c in zip(values, cls):
        yield func, v, c


def test_power_rule_comp():
    ops = (ReciprocalOperator(), SqrtOperator(), SquareOperator(),
           PowerOperator(2.5))
    op = CompositionOperator(ops)
    assert_is_type(op, PowerOperator)
    assert_equal(op.n, -2.5)


def test_power_rule_mul():
    ops = (ReciprocalOperator(), SqrtOperator(), SquareOperator(),
           PowerOperator(2.5))
    op = MultiplicationOperator(ops)
    assert_is_type(op, PowerOperator)
    assert_equal(op.n, 4)


def test_hard_thresholding():
    x = [-1., -0.2, -0.1, 0, 0.2, 0.1, 2, 3]
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
    K = G(H)
    assert_is_instance(K, HardThresholdingOperator)
    assert_equal(K.a, np.maximum(lbda, lbda2))
    assert_equal(K.shapein, shape)
    K = H(G)
    assert_is_instance(K, HardThresholdingOperator)
    assert_equal(K.a, np.maximum(lbda, lbda2))
    assert_equal(K.shapein, shape)

    H = HardThresholdingOperator([0, 0])
    assert_is_instance(H, IdentityOperator)
    assert_equal(H.shapein, (2,))

    H = HardThresholdingOperator(0)
    assert_is_instance(H, IdentityOperator)
    assert H.flags.square
    assert_equal(H.flags.shape_input, 'implicit')
    assert_equal(H.flags.shape_output, 'implicit')


def test_soft_thresholding():
    x = [-1., -0.2, -0.1, 0, 0.1, 0.2, 2, 3]
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

    S = SoftThresholdingOperator([0, 0])
    assert_is_instance(S, IdentityOperator)
    assert_equal(S.shapein, (2,))

    S = SoftThresholdingOperator(0)
    assert_is_instance(S, IdentityOperator)
    assert S.flags.square
    assert_equal(S.flags.shape_input, 'implicit')
    assert_equal(S.flags.shape_output, 'implicit')
