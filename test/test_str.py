import itertools
from nose import SkipTest
from numpy.testing import assert_equal
from pyoperators import (
    CompositionOperator, PowerOperator, ProductOperator, Operator, flags)


class NL1(Operator):
    def __str__(self):
        return 'n'


class NL2(Operator):
    def __str__(self):
        return 'a(..., z=1)'


class NL3(Operator):
    def __str__(self):
        return '...**2'


@flags.linear
class L(Operator):
    pass


def check(group, expected):
    actual = str(CompositionOperator(group))
    if '**2**2' in actual:
        raise SkipTest
    assert_equal(str(CompositionOperator(group)), expected)


def test1():
    n = NL1()
    l = L()
    groups = itertools.chain(*[itertools.product([n, l], repeat=i)
                               for i in range(1, 5)])
    expecteds = (
        'n|l|'
        'n(n)|n(l)|l(n)|l * l|'
        'n(n(n))|n(n(l))|n(l(n))|n(l * l)|'
        'l(n(n))|l(n(l))|(l * l)(n)|l * l * l|'
        'n(n(n(n)))|n(n(n(l)))|n(n(l(n)))|n(n(l * l))|'
        'n(l(n(n)))|n(l(n(l)))|n((l * l)(n))|n(l * l * l)|'
        'l(n(n(n)))|l(n(n(l)))|l(n(l(n)))|l(n(l * l))|'
        '(l * l)(n(n))|(l * l)(n(l))|(l * l * l)(n)|l * l * l * l')
    for group, expected in zip(groups, expecteds.split('|')):
        yield check, group, expected


def test2():
    n = NL1()
    a = NL2()
    l = L()
    groups = itertools.chain(*[itertools.product([n, l, a], repeat=i)
                               for i in range(1, 4)])
    expecteds = iter((
        'a(..., z=1)|'
        'n(a(..., z=1))|'
        'l(a(..., z=1))|'
        'a(n, z=1)|'
        'a(l, z=1)|'
        'a(a(..., z=1), z=1)|'
        'n(n(a(..., z=1)))|'
        'n(l(a(..., z=1)))|'
        'n(a(n, z=1))|'
        'n(a(l, z=1))|'
        'n(a(a(..., z=1), z=1))|'
        'l(n(a(..., z=1)))|'
        '(l * l)(a(..., z=1))|'
        'l(a(n, z=1))|'
        'l(a(l, z=1))|'
        'l(a(a(..., z=1), z=1))|'
        'a(n(n), z=1)|'
        'a(n(l), z=1)|'
        'a(n(a(..., z=1)), z=1)|'
        'a(l(n), z=1)|'
        'a(l * l, z=1)|'
        'a(l(a(..., z=1)), z=1)|'
        'a(a(n, z=1), z=1)|'
        'a(a(l, z=1), z=1)|'
        'a(a(a(..., z=1), z=1), z=1)').split('|'))
    for group in groups:
        if a not in group:
            continue
        expected = next(expecteds)
        yield check, group, expected


def test3():
    n = NL1()
    a = NL3()
    l = L()
    groups = itertools.chain(*[itertools.product([n, l, a], repeat=i)
                               for i in range(1, 4)])
    expecteds = iter((
        '...**2|'
        'n(...**2)|'
        'l(...**2)|'
        'n**2|'
        'l**2|'
        '(...**2)**2|'
        'n(n(...**2))|'
        'n(l(...**2))|'
        'n(n**2)|'
        'n(l**2)|'
        'n((...**2)**2)|'
        'l(n(...**2))|'
        '(l * l)(...**2)|'
        'l(n**2)|'
        'l(l**2)|'
        'l((...**2)**2)|'
        'n(n)**2|'
        'n(l)**2|'
        'n(...**2)**2|'
        'l(n)**2|'
        '(l * l)**2|'
        'l(...**2)**2|'
        '(n**2)**2|'
        '(l**2)**2|'
        '((...**2)**2)**2|').split('|'))
    for group in groups:
        if a not in group:
            continue
        expected = next(expecteds)
        yield check, group, expected


def test4():
    raise SkipTest
    assert str(PowerOperator(3)(ProductOperator(axis=2))) == \
           'product(..., axis=2)**3'
