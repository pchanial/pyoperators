from __future__ import division
from numpy.testing import assert_equal, assert_raises
from pyoperators.brackets import bracket, BracketList

brackets = (
    bracket(1, 2),
    bracket(1, 3),
    bracket(1, 1),
    bracket(0, 1),
    bracket(3, 3),
    bracket(-1, 2),
)


def test_bracket_len():
    expecteds = (2, 3, 1, 2, 1, 4)

    def func(b, e):
        assert_equal(len(b), e)

    for b, e in zip(brackets, expecteds):
        yield func, b, e


def test_bracket_cmp():
    sbrackets = sorted(brackets)
    expecteds = (-1, 2), (0, 1), (1, 1), (1, 2), (1, 3), (3, 3)

    def func(b, e):
        assert_equal(b.start, e[0])
        assert_equal(b.stop, e[1])

    for b, e in zip(sbrackets, expecteds):
        yield func, b, e


def test_bracket_contains():
    brackets1 = (bracket(0, 0), bracket(0, 1))
    brackets2 = (
        bracket(0, 0),
        bracket(0, 1),
        bracket(1, 1),
        bracket(0, 2),
        bracket(1, 2),
    )
    expecteds = ((True, True, False, True, False), (False, True, False, True, False))

    def func(b1, b2, e):
        assert_equal(b1 in b2, e)

    for b1, expected in zip(brackets1, expecteds):
        for b2, e in zip(brackets2, expected):
            yield func, b1, b2, e


def test_bracket_intersection():
    brackets1 = bracket(-2, 0), bracket(-1, 0), bracket(0, 0)
    brackets2 = bracket(-1, 2), bracket(0, 2), bracket(0, 1), bracket(0, 0)
    expecteds = (
        (bracket(-1, 0), bracket(0, 0), bracket(0, 0), bracket(0, 0)),
        (bracket(-1, 0), bracket(0, 0), bracket(0, 0), bracket(0, 0)),
        (bracket(0, 0), bracket(0, 0), bracket(0, 0), bracket(0, 0)),
    )

    def func(b1, b2, e):
        b1 = b1.copy()
        b1.intersection_update(b2)
        assert_equal(b1, e)

    for b1, expected in zip(brackets1, expecteds):
        for b2, e in zip(brackets2, expected):
            yield func, b1, b2, e

    brackets2 = bracket(1, 1), bracket(1, 1), bracket(2, 2)

    def func(b1, b2):
        assert_raises(IndexError, b1.intersection_update, b2)

    for b1 in brackets1:
        for b2 in brackets2:
            yield func, b1, b2


def test_list():
    bracketlists = (
        BracketList(),
        BracketList([bracket(1, 2)]),
        BracketList([bracket(1, 4), bracket(3, 7)]),
        BracketList([bracket(1, 4), bracket(3, 6), bracket(7, 7)]),
        BracketList([bracket(1, 1), bracket(2, 7), bracket(5, 7)]),
        BracketList([bracket(1, 4), bracket(3, 7), bracket(7, 7)]),
        BracketList([bracket(1, 1), bracket(1, 7), bracket(5, 7)]),
        BracketList([bracket(1, 2), bracket(2, 3), bracket(3, 4)]),
    )
    expecteds = (
        ([],),
        ([bracket(1, 2)],),
        ([bracket(3, 4)],),
        ([bracket(3, 4), bracket(7, 7)],),
        ([bracket(1, 1), bracket(5, 7)],),
        ([bracket(1, 4), bracket(7, 7)],),
        ([bracket(1, 1), bracket(5, 7)],),
        ([bracket(2, 2), bracket(3, 4)], [bracket(1, 2), bracket(3, 3)]),
    )

    def func(bracketlist, expected):
        assert_equal(bracketlist.disjoint_intersections(), expected)

    for bracketlist, expected in zip(bracketlists, expecteds):
        yield func, bracketlist, expected
