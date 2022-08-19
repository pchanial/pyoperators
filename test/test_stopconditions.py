import itertools

import pytest

from pyoperators.iterative.stopconditions import StopCondition


class A:
    pass


sc1 = StopCondition(lambda _: _.a > 2, 'a>2')
sc2 = StopCondition(lambda _: _.b > 2, 'b>2')
sc3 = StopCondition(lambda _: _.c > 2, 'c>2')
sc = sc1 or sc2 or sc2


@pytest.mark.parametrize('value', [1, 3])
def test_stop_condition(value):
    class A:
        def __init__(self, a):
            self.a = a

    obj = A(value)
    if value > 2:
        with pytest.raises(StopIteration):
            sc1(obj)
    else:
        sc1(obj)


@pytest.mark.parametrize('a, b, c', itertools.product((1, 3), repeat=3))
def test_stop_condition_or(a, b, c):
    class ABC:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    obj = ABC(a, b, c)
    if any(_ > 2 for _ in (a, b, c)):
        try:
            sc(obj)
        except StopIteration as e:
            if obj.a > 2:
                assert str(e) == str(sc1)
            elif obj.b > 2:
                assert str(e) == str(sc2)
            else:
                assert str(e) == str(sc3)
