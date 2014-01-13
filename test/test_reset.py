from __future__ import division
from numpy.testing import assert_equal
from pyoperators import Operator
from .common import OPS


def test():
    def func(Op):
        op = Op()
        op._reset(shapein=3)
        assert_equal(op.flags.shape_input, 'explicit')
        op = Op()
        op._reset(shapeout=3)
        assert_equal(op.flags.shape_output, 'explicit')
    for Op in OPS:
        yield func, Op


def test_square():
    op = Operator(shapein=3, shapeout=3)
    assert op.flags.square
    op._reset(shapeout=4)
    assert not op.flags.square

    op = Operator(shapein=3, shapeout=3)
    op._reset(shapein=4)
    assert not op.flags.square
