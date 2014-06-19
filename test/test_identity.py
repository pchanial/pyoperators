from pyoperators import IdentityOperator
from pyoperators.utils.testing import (
    assert_eq, assert_is, assert_is_type)
from .common import OPS, ndarray2, attr2
ops = [_() for _ in OPS] + [_(flags={'linear': False,
                                     'contiguous_input': True}) for _ in OPS]


def test_rule_right():
    ids = (IdentityOperator(classout=ndarray2, attrout=attr2),
           IdentityOperator(shapein=4, classout=ndarray2, attrout=attr2))

    def func(id_, op_):
        op = id_(op_)
        assert_is_type(op, type(op_))
        attr = {}
        assert_is(op.classout, id_.classout)
        attr.update(op_.attrout)
        attr.update(id_.attrout)
        assert_eq(op.attrout, attr)
        assert_eq(op.flags.linear, op_.flags.linear)
        assert_eq(op.flags.contiguous_input, op_.flags.contiguous_input)
    for id_ in ids:
        for op_ in ops:
            yield func, id_, op_


def test_rule_left():
    ids = (IdentityOperator(classout=ndarray2, attrout=attr2),
           IdentityOperator(shapein=3, classout=ndarray2, attrout=attr2))

    def func(op_, id_):
        op = op_(id_)
        assert_is_type(op, type(op_))
        attr = {}
        assert_is(op.classout, op_.classout)
        attr.update(id_.attrout)
        attr.update(op_.attrout)
        assert_eq(op.attrout, attr)
        assert_eq(op.flags.linear, op_.flags.linear)
        assert_eq(op.flags.contiguous_input, op_.flags.contiguous_input)
    for op_ in ops:
        for id_ in ids:
            yield func, op_, id_
