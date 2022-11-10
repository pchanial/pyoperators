import pytest

from pyoperators import IdentityOperator
from pyoperators.utils.testing import assert_is_type

from .common import OPS, attr2, ndarray2

ops = [_() for _ in OPS] + [
    _(flags={'linear': False, 'contiguous_input': True}) for _ in OPS
]


@pytest.mark.parametrize(
    'id_',
    [
        IdentityOperator(classout=ndarray2, attrout=attr2),
        IdentityOperator(shapein=4, classout=ndarray2, attrout=attr2),
    ],
)
@pytest.mark.parametrize('op_', ops)
def test_rule_right(id_, op_):
    op = id_ @ op_
    assert type(op) is type(op_)
    attr = {}
    assert op.classout is id_.classout
    attr.update(op_.attrout)
    attr.update(id_.attrout)
    assert op.attrout == attr
    assert op.flags.linear == op_.flags.linear
    assert op.flags.contiguous_input == op_.flags.contiguous_input


@pytest.mark.parametrize(
    'id_',
    [
        IdentityOperator(classout=ndarray2, attrout=attr2),
        IdentityOperator(shapein=3, classout=ndarray2, attrout=attr2),
    ],
)
@pytest.mark.parametrize('op_', ops)
def test_rule_left(id_, op_):
    op = op_ @ id_
    assert_is_type(op, type(op_))
    attr = {}
    assert op.classout is op_.classout
    attr.update(id_.attrout)
    attr.update(op_.attrout)
    assert op.attrout == attr
    assert op.flags.linear == op_.flags.linear
    assert op.flags.contiguous_input == op_.flags.contiguous_input
