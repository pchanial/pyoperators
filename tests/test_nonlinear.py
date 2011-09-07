import nose
import numpy as np
from numpy.testing import assert_array_equal

from operators import RoundOperator


def test_rounding():
    a = np.array([-3.5, -3, -2.6, -2.5, -2.4, 0, 0.2, 0.5, 0.9, 1, 1.5])
    r = RoundOperator('rtz')
    assert_array_equal(r(a), [-3, -3, -2, -2, -2, 0, 0, 0, 0, 1, 1])
    # r = RoundOperator('rti')
    # assert_array_equal(r(a), [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2])
    r = RoundOperator('rtmi')
    assert_array_equal(r(a), [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1])
    r = RoundOperator('rtpi')
    assert_array_equal(r(a), [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2])
    # r = RoundOperator('rhtz')
    # assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1])
    # r = RoundOperator('rhti')
    # assert_array_equal(r(a), [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2])
    # r = RoundOperator('rhtmi')
    # assert_array_equal(r(a), [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1])
    # r = RoundOperator('rhtpi')
    # assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2])
    r = RoundOperator('rhte')
    assert_array_equal(r(a), [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2])
    # r = RoundOperator('rhto')
    # assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1])
    # r = RoundOperator('rhs')
    # mask = np.array([True,True,False,True,True,True,False,True,True], np.bool)
    # result = r(a)
    # assert_array_equal(result[mask], [-3,-3,-2,0,0,1,1])
    # assert result[2] in (-3,-2)
    # assert result[-3] in (0,1)


if __name__ == "__main__":
    nose.run(argv=['', __file__])
