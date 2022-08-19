import numpy as np
import pytest
from numpy.testing import assert_equal

import pyoperators
from pyoperators.iterative import criterions

sizes = [1, 4, 16, 100]
values = [-10, -1, 0, 2]


@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('value', values)
def test_norm2(size, value):
    N = criterions.Norm2()
    assert N(value * np.ones(size)) == size * value**2


@pytest.mark.parametrize('factor', [-2.0, -1, 0, 1.0, 2.0])
@pytest.mark.parametrize('value', values)
def test_norm2_mul(factor, value):
    N = criterions.Norm2()
    N2 = factor * N
    vec = value * np.ones(1)
    assert factor * N(vec) == N2(vec)


@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('value', values)
def test_dnorm2(size, value):
    N = criterions.Norm2()
    assert_equal(N.diff(value * np.ones(size)), 2 * value * np.ones(size))


@pytest.mark.parametrize('factor', [-2.0, -1, 0, 1.0, 2.0])
@pytest.mark.parametrize('value', values)
def test_dnorm2_mul(factor, value):
    N = criterions.Norm2()
    N2 = factor * N
    vec = value * np.ones(1)
    assert_equal(factor * N.diff(vec), N2.diff(vec))


@pytest.mark.parametrize('shapein', [(1,), (2,), (2, 3)])
def test_elements(shapein):
    N = criterions.Norm2()
    I = pyoperators.IdentityOperator(shapein=shapein)
    C0 = criterions.CriterionElement(N, I)
    assert C0(np.ones(shapein)) == np.prod(shapein)
