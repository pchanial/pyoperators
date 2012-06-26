from __future__ import division

import numpy as np
from pyoperators.linear import PackOperator, UnpackOperator, SumOperator
from pyoperators.utils.testing import assert_eq

SHAPES = (None, (), (1,), (3,), (2,3), (2,3,4))

def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1,2,3,4,5]), [1,4])
    assert np.allclose(p.T([1,4]), [1,0,0,4,0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1,4]), [1,0,0,4,0])
    assert np.allclose(u.T([1,2,3,4,5]), [1,4])

def test_sum_operator():
    for s in SHAPES[2:]:
        for a in [None] + list(range(len(s))):
            op = SumOperator(axis=a)
            d = op.todense(shapein=s)
            t = op.T.todense(shapeout=s)
            assert_eq(d, t.T)
