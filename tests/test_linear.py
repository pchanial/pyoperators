from __future__ import division

import numpy as np
from pyoperators import PackOperator, UnpackOperator

def test_packing():

    p = PackOperator([False, True, True, False, True])
    assert p.T.__class__ == UnpackOperator
    assert np.allclose(p([1,2,3,4,5]), [1,4])
    assert np.allclose(p.T([1,4]), [1,0,0,4,0])

    u = UnpackOperator([False, True, True, False, True])
    assert u.T.__class__ == PackOperator
    assert np.allclose(u([1,4]), [1,0,0,4,0])
    assert np.allclose(u.T([1,2,3,4,5]), [1,4])
