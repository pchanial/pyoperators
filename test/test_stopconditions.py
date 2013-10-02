from __future__ import division

import itertools

from pyoperators.iterative.stopconditions import StopCondition
from pyoperators.utils.testing import assert_eq, assert_raises


class A():
    pass

sc1 = StopCondition(lambda s: s.a > 2, 'a>2')
sc2 = StopCondition(lambda s: s.b > 2, 'b>2')
sc3 = StopCondition(lambda s: s.c > 2, 'c>2')


def test_stop_condition():
    values = (1, 3)

    def func(v):
        a = A()
        a.a = v
        if v > 2:
            assert_raises(StopIteration, sc1, a)
    for v in values:
        yield func, v


def test_stop_condition_or():
    sc = sc1 or sc2 or sc2

    def func(v):
        a = A()
        a.a, a.b, a.c = v
        if any(_ > 2 for _ in v):
            try:
                sc(a)
            except StopIteration as e:
                if a.a > 2:
                    assert_eq(str(e), str(sc1))
                elif a.b > 2:
                    assert_eq(str(e), str(sc2))
                else:
                    assert_eq(str(e), str(sc3))
    for v in itertools.product((1, 3), repeat=3):
        yield func, v
