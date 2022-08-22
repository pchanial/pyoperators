import pytest

from pyoperators import Operator

from .common import OPS


@pytest.mark.parametrize('cls', OPS)
def test(cls):
    op = cls()
    op._reset(shapein=3)
    assert op.flags.shape_input == 'explicit'

    op = cls()
    op._reset(shapeout=3)
    assert op.flags.shape_output == 'explicit'


def test_square():
    op = Operator(shapein=3, shapeout=3)
    assert op.flags.square

    op._reset(shapeout=4)
    assert not op.flags.square

    op = Operator(shapein=3, shapeout=3)
    op._reset(shapein=4)
    assert not op.flags.square
