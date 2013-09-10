#!/usr/bin/env python
import nose
import numpy as np
from numpy import testing

import pyoperators
from pyoperators.iterative import criterions

sizes = (1, 4, 16, 100)
values = (-10, -1, 0, 2)
alist = (-2., -1, 0, 1., 2.)
shapeins = ((1, ), (2, ), (2, 3))


# norms
def check_norm2(size, value):
    N = criterions.Norm2()
    assert N(value * np.ones(size)) == size * value ** 2


def test_norm2():
    for size in sizes:
        for value in values:
            yield check_norm2, size, value


def check_dnorm2(size, value):
    N = criterions.Norm2()
    testing.assert_array_equal(N.diff(value * np.ones(size)), 2 * value * np.ones(size))


def test_dnorm2():
    for size in sizes:
        for value in values:
            yield check_dnorm2, size, value


def check_norm2_mul(a, value):
    N = criterions.Norm2()
    N2 = a * N
    vec = value * np.ones(1)
    assert a * N(vec) == N2(vec)


def test_norm2_mul():
    for a in alist:
        for value in values:
            yield check_norm2_mul, a, value


def check_dnorm2_mul(a, value):
    N = criterions.Norm2()
    N2 = a * N
    vec = value * np.ones(1)
    testing.assert_array_equal(a * N.diff(vec), N2.diff(vec))


def test_dnorm2_mul():
    for a in alist:
        for value in values:
            yield check_dnorm2_mul, a, value


# criterion elements
def check_elements(shapein):
    N = criterions.Norm2()
    I = pyoperators.IdentityOperator(shapein=shapein)
    C0 = criterions.CriterionElement(N, I)
    assert C0(np.ones(shapein)) == np.prod(shapein)


def test_elements():
    for shapein in shapeins:
        yield check_elements, shapein
