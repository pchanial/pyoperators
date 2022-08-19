from itertools import chain, repeat

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from pyoperators import (
    Cartesian2SphericalOperator,
    CompositionOperator,
    ConstantOperator,
    HardThresholdingOperator,
    IdentityOperator,
    MultiplicationOperator,
    NormalizeOperator,
    NumexprOperator,
    PowerOperator,
    ReciprocalOperator,
    RoundOperator,
    SoftThresholdingOperator,
    Spherical2CartesianOperator,
    SqrtOperator,
    SquareOperator,
)
from pyoperators.utils import product as reduce_product
from pyoperators.utils.testing import assert_same


@pytest.mark.parametrize('convention', Cartesian2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize(
    'coords, shape',
    [
        ((1, 0, 0), ()),
        ((0, 1, 0), ()),
        ((0, 0, 1), ()),
        (((1, 0, 0), (0, 1, 0), (0, 0, 1)), (3,)),
        ((((1, 0, 0), (0, 1, 0)),), (1, 2)),
    ],
)
@pytest.mark.parametrize('degrees', [False, True])
def test_cartesian_spherical(convention, coords, shape, degrees):
    c2s = Cartesian2SphericalOperator(convention, degrees=degrees)
    s2c = Spherical2CartesianOperator(convention, degrees=degrees)
    a = s2c(c2s(coords))
    assert a.shape == shape + (3,)
    assert_allclose(a, coords, atol=1e-16)


@pytest.mark.parametrize(
    'value',
    [3, pytest.param('bla', marks=pytest.mark.xfail)],  # FIXME: ValueError is raised.
)
def test_cartesian_spherical_wrong_type(value):
    with pytest.raises(TypeError):
        Cartesian2SphericalOperator(value)


@pytest.mark.parametrize('input', [np.array(1.0), np.zeros(2), np.zeros(3)])
@pytest.mark.parametrize('output', [np.array(1.0), np.zeros(2), np.zeros(3)])
def test_cartesian_spherical_wrong_shape(input, output):
    op = Cartesian2SphericalOperator('zenith,azimuth')
    if input.shape == (3,) and output.shape == (2,):
        op(input, output)
        return
    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('convention1', ['zenith,azimuth', 'azimuth,elevation'])
@pytest.mark.parametrize('convention2', ['zenith,azimuth', 'azimuth,elevation'])
def test_cartesian_spherical_rules(convention1, convention2):
    op1 = Cartesian2SphericalOperator(convention1)
    assert op1.convention == convention1
    assert type(op1.I) is Spherical2CartesianOperator

    op2 = Spherical2CartesianOperator(convention2)
    assert op2.convention == convention2
    assert type(op2.I) is Cartesian2SphericalOperator

    op = op1(op2)
    if convention1 == convention2:
        assert type(op) is IdentityOperator
    else:
        assert type(op) is CompositionOperator


DIRS_ZA = (
    (0, 0),
    (20, 0),
    (130, 0),
    (10, 20),
    (20, 190),
    ((0, 0), (20, 0), (130, 0), (10, 20), (20, 130)),
    (((0, 0), (20, 200), (130, 300)),),
)
DIRS_AZ = (
    (0, 0),
    (0, 20),
    (0, 130),
    (20, 10),
    (190, 20),
    ((0, 0), (0, 20), (0, 130), (20, 10), (130, 20)),
    (((0, 0), (200, 20), (300, 130)),),
)
DIRS_EA = (
    (90, 0),
    (70, 0),
    (-40, 0),
    (80, 20),
    (70, 190),
    ((90, 0), (70, 0), (-40, 0), (80, 20), (70, 130)),
    (((90, 0), (70, 200), (-40, 300)),),
)
DIRS_AE = (
    (0, 90),
    (0, 70),
    (0, -40),
    (20, 80),
    (190, 70),
    ((0, 90), (0, 70), (0, -40), (20, 80), (130, 70)),
    (((0, 90), (200, 70), (300, -40)),),
)
SHAPES_TEST_SC = ((), (), (), (), (), (5,), (1, 3))
REFS_TEST_SC = [
    Spherical2CartesianOperator('zenith,azimuth')(np.radians(v)) for v in DIRS_ZA
]


@pytest.mark.parametrize(
    'convention, vec, shape, ref',
    chain(
        zip(repeat('zenith,azimuth'), DIRS_ZA, SHAPES_TEST_SC, REFS_TEST_SC),
        zip(repeat('azimuth,zenith'), DIRS_AZ, SHAPES_TEST_SC, REFS_TEST_SC),
        zip(repeat('elevation,azimuth'), DIRS_EA, SHAPES_TEST_SC, REFS_TEST_SC),
        zip(repeat('azimuth,elevation'), DIRS_AE, SHAPES_TEST_SC, REFS_TEST_SC),
    ),
)
@pytest.mark.parametrize('degrees', [False, True])
def test_spherical_cartesian(convention, vec, shape, ref, degrees):
    orig = vec
    if not degrees:
        vec = np.radians(vec)
    s2c = Spherical2CartesianOperator(convention, degrees=degrees)
    c2s = Cartesian2SphericalOperator(convention, degrees=degrees)
    assert_allclose(s2c(vec), ref)
    a = c2s(s2c(vec))
    if not degrees:
        a = np.degrees(a)
    assert a.shape == shape + (2,)
    assert_allclose(a, orig, atol=1e-16)


@pytest.mark.parametrize('value', [3, pytest.param('bla', marks=pytest.mark.xfail)])
def test_spherical_cartesian_wrong_type(value):
    with pytest.raises(TypeError):
        Spherical2CartesianOperator(value)


@pytest.mark.parametrize('input', [np.array(1.0), np.zeros(2), np.zeros(3)])
@pytest.mark.parametrize('output', [np.array(1.0), np.zeros(2), np.zeros(3)])
def test_spherical_cartesian_wrong_shape(input, output):
    op = Spherical2CartesianOperator('zenith,azimuth')
    if input.shape == (2,) and output.shape == (3,):
        op(input, output)
        return
    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('convention1', ['zenith,azimuth', 'azimuth,elevation'])
@pytest.mark.parametrize('convention2', ['zenith,azimuth', 'azimuth,elevation'])
def test_spherical_cartesian_rules(convention1, convention2):
    op1 = Spherical2CartesianOperator(convention1)
    assert op1.convention == convention1
    assert type(op1.I) is Cartesian2SphericalOperator

    op2 = Cartesian2SphericalOperator(convention2)
    assert op2.convention == convention2
    assert type(op2.I) is Spherical2CartesianOperator

    op = op1(op2)
    if convention1 == convention2:
        assert type(op) is IdentityOperator
    else:
        assert type(op) is CompositionOperator


ROUNDING_VALUES = [-3.5, -3, -2.6, -2.5, -2.4, 0, 0.2, 0.5, 0.9, 1, 1.5]


@pytest.mark.parametrize(
    'method, expected_values',
    [
        ('rtz', [-3, -3, -2, -2, -2, 0, 0, 0, 0, 1, 1]),
        ('rti', [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2]),
        ('rtmi', [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1]),
        ('rtpi', [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2]),
        ('rhtz', [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]),
        ('rhti', [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2]),
        ('rhtmi', [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1]),
        ('rhtpi', [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2]),
        ('rhte', [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2]),
        ('rhto', [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]),
    ],
)
def test_rounding(method, expected_values):
    if method in {'rti', 'rhtz', 'rhti', 'rhto'}:
        pytest.xfail(f'Rounding with method {method!r} is not implemented.')
    op = RoundOperator(method)
    assert_equal(op(ROUNDING_VALUES), expected_values)


def test_rounding_rhs():
    pytest.xfail("Rounding with method 'rhs' is not implemented.")
    r = RoundOperator('rhs')
    mask = np.array([True, True, False, True, True, True, False, True, True], np.bool)
    result = r(ROUNDING_VALUES)
    assert_equal(result[mask], [-3, -3, -2, 0, 0, 1, 1])
    assert result[2] in (-3, -2)
    assert result[-4] in (0, 1)


@pytest.mark.parametrize('shape', [(2,), (4,), (2, 3), (4, 5, 2)])
def test_normalize(shape):
    n = NormalizeOperator()
    vec = np.arange(reduce_product(shape)).reshape(shape)
    exp = vec / np.sqrt(np.sum(vec**2, axis=-1))[..., None]
    assert_same(n(vec), exp)


def test_numexpr1():
    d = 7.0
    op = NumexprOperator('2.*exp(input)+d', {'d': d})
    assert_allclose(op(3.0), 2 * np.exp(3.0) + d)


def test_numexpr2():
    op = NumexprOperator('3*input') + NumexprOperator('2*input')
    assert_equal(op(np.arange(10)), 5 * np.arange(10))


@pytest.mark.parametrize(
    'value, expected_cls',
    [
        (-1, ReciprocalOperator),
        (0, ConstantOperator),
        (0.5, SqrtOperator),
        (1, IdentityOperator),
        (2, SquareOperator),
        (3, PowerOperator),
    ],
)
def test_power(value, expected_cls):
    op = PowerOperator(value)
    assert type(op) is expected_cls
    if isinstance(op, PowerOperator):
        assert op.n == value


def test_power_rule_comp():
    ops = (ReciprocalOperator(), SqrtOperator(), SquareOperator(), PowerOperator(2.5))
    op = CompositionOperator(ops)
    assert type(op) is PowerOperator
    assert op.n == -2.5


def test_power_rule_mul():
    ops = (ReciprocalOperator(), SqrtOperator(), SquareOperator(), PowerOperator(2.5))
    op = MultiplicationOperator(ops)
    assert type(op) is PowerOperator
    assert op.n == 4


def test_hard_thresholding():
    x = [-1.0, -0.2, -0.1, 0, 0.2, 0.1, 2, 3]
    lbda = 0.2
    H = HardThresholdingOperator(lbda)
    expected = [-1, 0, 0, 0, 0, 0, 2, 3]
    assert_allclose(H(x), expected)

    x = np.array(x)
    H(x, x)
    assert_allclose(x, expected)

    lbda2 = [0.3, 0.1, 2]
    shape = np.asarray(lbda2).shape
    G = HardThresholdingOperator(lbda2)
    assert G.shapein == shape

    K = G(H)
    assert isinstance(K, HardThresholdingOperator)
    assert_allclose(K.a, np.maximum(lbda, lbda2))
    assert K.shapein == shape

    K = H(G)
    assert isinstance(K, HardThresholdingOperator)
    assert_allclose(K.a, np.maximum(lbda, lbda2))
    assert K.shapein == shape

    H = HardThresholdingOperator([0, 0])
    assert isinstance(H, IdentityOperator)
    assert H.shapein == (2,)

    H = HardThresholdingOperator(0)
    assert isinstance(H, IdentityOperator)
    assert H.flags.square
    assert H.flags.shape_input == 'implicit'
    assert H.flags.shape_output == 'implicit'


def test_soft_thresholding():
    x = [-1.0, -0.2, -0.1, 0, 0.1, 0.2, 2, 3]
    lbda = np.array(0.2)
    S = SoftThresholdingOperator(lbda)
    expected = [-1, 0, 0, 0, 0, 0, 2, 3] - lbda * [-1, 0, 0, 0, 0, 0, 1, 1]
    assert_allclose(S(x), expected)

    x = np.array(x)
    S(x, x)
    assert_allclose(x, expected)

    lbda2 = [0.3, 0.1, 2]
    shape = np.asarray(lbda2).shape
    T = SoftThresholdingOperator(lbda2)
    assert T.shapein == shape

    S = SoftThresholdingOperator([0, 0])
    assert isinstance(S, IdentityOperator)
    assert S.shapein == (2,)

    S = SoftThresholdingOperator(0)
    assert isinstance(S, IdentityOperator)
    assert S.flags.square
    assert S.flags.shape_input == 'implicit'
    assert S.flags.shape_output == 'implicit'
