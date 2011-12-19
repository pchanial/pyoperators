import nose
import numpy as np
from numpy.testing import assert_, assert_array_equal

from pyoperators import NumexprOperator, RoundOperator


def test_rounding():
    a = np.array([-3.5, -3, -2.6, -2.5, -2.4, 0, 0.2, 0.5, 0.9, 1, 1.5])
    r = RoundOperator('rtz')
    yield assert_array_equal, r(a), [-3, -3, -2, -2, -2, 0, 0, 0, 0, 1, 1]
    # r = RoundOperator('rti')
    # yield assert_array_equal, r(a), [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2]
    r = RoundOperator('rtmi')
    yield assert_array_equal, r(a), [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1]
    r = RoundOperator('rtpi')
    yield assert_array_equal, r(a), [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2]
    # r = RoundOperator('rhtz')
    # yield assert_array_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    # r = RoundOperator('rhti')
    # yield assert_array_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2]
    # r = RoundOperator('rhtmi')
    # yield assert_array_equal, r(a), [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1]
    # r = RoundOperator('rhtpi')
    # yield assert_array_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2]
    r = RoundOperator('rhte')
    yield assert_array_equal, r(a), [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2]
    # r = RoundOperator('rhto')
    # yield assert_array_equal, r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1]
    # r = RoundOperator('rhs')
    # mask = np.array([True,True,False,True,True,True,False,True,True], np.bool)
    # result = r(a)
    # yield assert_array_equal, result[mask], [-3,-3,-2,0,0,1,1]
    # yield assert_, result[2] in (-3,-2)
    # yield assert_, result[-4] in (0,1)


def test_numexpr():
    d = 7.0
    op = NumexprOperator('2.*exp(input)+d', {'d': d})
    assert op(3.0) == 2 * np.exp(3.0) + d


if __name__ == "__main__":
    nose.run(argv=['', __file__])
