import nose
import numpy as np
from numpy.testing import assert_array_equal

from operators import RoundingOperator

def test_rounding():
    a=np.array([-3.5,-3,-2.6,-2.5,-2.4,0,0.2,0.5,0.9,1,1.5])
    r = RoundingOperator('rtz')
    assert_array_equal(r(a), [-3,-3, -2, -2, -2, 0, 0, 0, 0, 1, 1])
    #r = RoundingOperator('rti')
    #assert_array_equal(r(a), [-4, -3, -3, -3, -3, 0, 1, 1, 1, 2])
    r = RoundingOperator('rtmi')
    assert_array_equal(r(a), [-4, -3, -3, -3, -3, 0, 0, 0, 0, 1, 1])
    r = RoundingOperator('rtpi')
    assert_array_equal(r(a), [-3, -3, -2, -2, -2, 0, 1, 1, 1, 1, 2])
    #r = RoundingOperator('rhtz')
    #assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1])
    #r = RoundingOperator('rhti')
    #assert_array_equal(r(a), [-4, -3, -3, -3, -2, 0, 0, 1, 1, 2])
    #r = RoundingOperator('rhtmi')
    #assert_array_equal(r(a), [-4, -3, -3, -3, -2, 0, 0, 0, 1, 1, 1])
    #r = RoundingOperator('rhtpi')
    #assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 1, 1, 1, 2])
    r = RoundingOperator('rhte')
    assert_array_equal(r(a), [-4, -3, -3, -2, -2, 0, 0, 0, 1, 1, 2])
    #r = RoundingOperator('rhto')
    #assert_array_equal(r(a), [-3, -3, -3, -2, -2, 0, 0, 0, 1, 1, 1])
    #r = RoundingOperator('rhs')
    #mask = np.array([True,True,False,True,True,True,False,True,True], np.bool)
    #result = r(a)
    #assert_array_equal(result[mask], [-3,-3,-2,0,0,1,1])
    #assert result[2] in (-3,-2)
    #assert result[-3] in (0,1)


if __name__ == "__main__":
    nose.run(argv=['', __file__])
