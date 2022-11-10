import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import (
    CompositionOperator,
    ConstantOperator,
    O,
    Operator,
    ZeroOperator,
    flags,
    rule_manager,
)
from pyoperators.utils import ndarraywrap
from pyoperators.utils.testing import assert_equal

from .common import OPS, attr2, ndarray2

op = Operator()
TEST_ZERO_OPS = [_() for _ in OPS] + [_(flags={'linear': False}) for _ in OPS]
ZERO_LEFTS = (
    ZeroOperator(classout=ndarray2, attrout=attr2),
    ZeroOperator(shapein=4, classout=ndarray2, attrout=attr2),
)
ZEROS_RIGHTS = (
    ZeroOperator(classout=ndarray2, attrout=attr2),
    ZeroOperator(classout=ndarray2, attrout=attr2, flags='square'),
    ZeroOperator(shapein=3, classout=ndarray2, attrout=attr2),
    ZeroOperator(shapein=3, shapeout=3, classout=ndarray2, attrout=attr2),
)


def test_zero1():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6)
    zo = z(o)
    assert isinstance(zo, ZeroOperator)
    assert zo.shapein == o.shapein
    assert zo.shapeout is None


def test_zero2():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator()
    zo = z(o)
    assert isinstance(zo, ZeroOperator)
    assert zo.shapein is None, 'in'
    assert zo.shapeout == z.shapeout, 'out'


def test_zero3():
    z = ZeroOperator(shapein=3, shapeout=6)
    o = Operator(flags='square')
    zo = z @ o
    assert isinstance(zo, ZeroOperator)
    assert zo.shapein == z.shapein, 'in'
    assert zo.shapeout == z.shapeout, 'out'


def test_zero4():
    z = ZeroOperator()
    o = Operator(flags='linear')
    assert isinstance(z @ o, ZeroOperator)
    assert isinstance(o @ z, ZeroOperator)


def test_zero5():
    z = ZeroOperator()
    o = Operator(shapein=3, shapeout=6, flags='linear')
    zo = z @ o
    oz = o @ z
    assert isinstance(zo, ZeroOperator), 'zo'
    assert zo.shapein == o.shapein, 'zo in'
    assert zo.shapeout is None, 'zo out'
    assert isinstance(oz, ZeroOperator), 'oz'
    assert oz.shapein is None, 'oz, in'
    assert oz.shapeout == o.shapeout, 'oz, out'


@pytest.mark.xfail(reason='reason: Unknown.')
def test_zero6():
    @flags.linear
    class Op(Operator):
        def direct(self, input, output):
            output[:] = np.concatenate([input, 2 * input])

        def transpose(self, input, output):
            output[:] = input[0 : output.size]

        def reshapein(self, shapein):
            return (2 * shapein[0],)

        def reshapeout(self, shapeout):
            return (shapeout[0] // 2,)

    z = ZeroOperator(flags='square')
    o = Op()
    od = o.todense(shapein=4)
    zo = z @ o
    zod_ref = np.dot(np.zeros((8, 8)), od)
    assert_equal((z @ o).todense(shapein=4), zod_ref)
    oz = o @ z
    ozd_ref = np.dot(od, np.zeros((4, 4)))
    assert_equal((o @ z).todense(shapein=4), ozd_ref)
    assert_equal(zo.T.todense(shapein=8), zod_ref.T)
    assert_equal(oz.T.todense(shapein=8), ozd_ref.T)


def test_zero7():
    z = ZeroOperator()
    assert z @ z == z


def test_zero8():
    class Op(Operator):
        pass

    o = Op()
    assert type(o + O) is Op


@pytest.mark.parametrize('op1', ZERO_LEFTS)
@pytest.mark.parametrize('op2', TEST_ZERO_OPS)
def test_merge_zero_left(op1, op2):
    op = op1(op2)
    assert isinstance(op, ZeroOperator)
    attr = {}
    attr.update(op2.attrout)
    attr.update(op1.attrout)
    assert op.attrout == attr
    x = np.ones(3)
    y = ndarraywrap(4)
    op(x, y)
    y2_tmp = np.empty(4)
    y2 = np.empty(4)
    op2(x, y2_tmp)
    op1(y2_tmp, y2)
    assert_equal(y, y2)
    assert isinstance(y, op1.classout)


@pytest.mark.parametrize('op1', TEST_ZERO_OPS)
@pytest.mark.parametrize('op2', ZEROS_RIGHTS)
def test_merge_zero_right(op1, op2):
    op = op1(op2)
    attr = {}
    attr.update(op2.attrout)
    attr.update(op1.attrout)
    assert op.attrout == attr
    assert op.classout is op1.classout
    if op1.flags.linear:
        assert type(op) is ZeroOperator
        assert_equal(op.todense(shapein=3, shapeout=4), np.zeros((4, 3)))
        return
    if (
        op1.flags.shape_output == 'unconstrained'
        or op1.flags.shape_input != 'explicit'
        and op2.flags.shape_output != 'explicit'
    ):
        assert type(op) is CompositionOperator
    else:
        assert type(op) is ConstantOperator

    if (
        op1.flags.shape_input == 'unconstrained'
        and op2.flags.shape_output == 'unconstrained'
    ):
        return
    with rule_manager(none=True):
        op_ref = op1(op2)
    assert_equal(
        op.todense(shapein=3, shapeout=4), op_ref.todense(shapein=3, shapeout=4)
    )
