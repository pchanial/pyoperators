import os
import tempfile
import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal

import pyoperators
from pyoperators import (
    AdditionOperator,
    CompositionOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
    MultiplicationOperator,
    Operator,
    PyOperatorsWarning,
)
from pyoperators.flags import linear
from pyoperators.rules import BinaryRule, RuleManager, UnaryRule, rule_manager
from pyoperators.utils import ndarraywrap

from .common import OPS, attr2, ndarray2

op = Operator()
TEST_RULE_OPS = [OP() for OP in OPS]

IDS_LEFTS = (
    IdentityOperator(classout=ndarray2, attrout=attr2),
    IdentityOperator(shapein=4, classout=ndarray2, attrout=attr2),
)
IDS_RIGHTS = (
    IdentityOperator(classout=ndarray2, attrout=attr2),
    IdentityOperator(shapein=3, classout=ndarray2, attrout=attr2),
)


class Operator1(Operator):
    pass


class Operator2(Operator):
    pass


class Operator3(Operator):
    pass


class Operator4(Operator1):
    pass


op1 = Operator1()
op2 = Operator2()
op3 = Operator3()
op4 = Operator4()


class NonLinearOperator(Operator):
    pass


@linear
class LinearOperator(Operator):
    pass


@linear
class AbsorbRightOperator(Operator):
    def __init__(self, value=3.0, **keywords):
        self.value = np.array(value)
        Operator.__init__(self, **keywords)
        self.set_rule(
            ('.', HomothetyOperator),
            lambda s, o: AbsorbRightOperator(s.value * o.data),
            CompositionOperator,
        )


@linear
class AbsorbLeftOperator(Operator):
    def __init__(self, value=3.0, **keywords):
        self.value = np.array(value)
        Operator.__init__(self, **keywords)
        self.set_rule(
            (HomothetyOperator, '.'),
            lambda o, s: AbsorbLeftOperator(s.value * o.data),
            CompositionOperator,
        )


nl = NonLinearOperator()
li = LinearOperator()
ar = AbsorbRightOperator()
al = AbsorbLeftOperator()
ho = HomothetyOperator(2)


def p1(o1, o2):
    return (o2, op1)


def p2(o1, o2):
    return op3


@pytest.mark.parametrize('subject', ['C', 'T', 'H', 'I', 'IC', 'IT', 'IH'])
def test_unaryrule_self(subject):
    r = UnaryRule(subject, '.')
    assert r(op1) is op1


@pytest.mark.parametrize('subject', ['C', 'T', 'H', 'I', 'IC', 'IT', 'IH'])
def test_unaryrule_identity(subject):
    r = UnaryRule(subject, '1')
    assert type(r(op1)) is IdentityOperator


@pytest.mark.parametrize(
    'subject, predicate',
    [
        ('.', '.'),
        ('.T', '.'),
        ('T', 'C'),
        ('T', 'T'),
        ('T', 'H'),
        ('T', 'I'),
    ],
)
def test_unaryrule_error(subject, predicate):
    with pytest.raises(ValueError):
        UnaryRule(subject, predicate)


@pytest.mark.parametrize('s1', ['.', 'C', 'T', 'H', 'I'])
@pytest.mark.parametrize('s2', ['.', 'C', 'T', 'H', 'I'])
@pytest.mark.parametrize('s3', ['1', '.', 'C', 'T', 'H', 'I'])
def test_binaryrule1(s1, s2, s3):
    if '.' not in (s1, s2):
        return

    op.T  # generate associated operators
    o1 = eval('op.' + s1) if s1 != '.' else op
    o2 = eval('op.' + s2) if s2 != '.' else op
    ref = o1 if s2[-1] != '.' else o2

    rule = BinaryRule(s1 + ',' + s2, s3)
    result = rule(o1, o2)
    assert result is not None
    assert isinstance(result, Operator)
    if s3 == '1':
        assert isinstance(result, IdentityOperator)
        return
    o3 = eval('ref.' + s3) if s3 != '.' else ref
    assert result is o3


def test_binaryrule2():
    rule = BinaryRule('.,T', p1)
    assert rule(op1, op2) is None
    assert rule(op1, op1.T) == (op1.T, op1)


def test_binaryrule3():
    rule = BinaryRule('.,T', p2)
    assert rule(op1, op2) is None
    assert isinstance(rule(op1, op1.T), Operator3)


def test_binaryrule4():
    rule = BinaryRule(('.', HomothetyOperator), p1)
    assert rule(op1, op2) is None
    s = HomothetyOperator(2)
    assert rule(op1, s) == (s, op1)


def test_binaryrule5():
    rule = BinaryRule((type(op1), '.'), p2)
    assert rule(op1, op1) == op3
    assert rule(op2, op1) is None
    assert rule(op4, op1) == op3


def test_binaryrule_priority():
    r1 = lambda o: None
    r2 = lambda o: None

    class Op1(Operator):
        pass

    class Op2(Op1):
        pass

    class Op3(Op2):
        def __init__(self):
            Op2.__init__(self)
            self.set_rule(('.', OpA), r1, CompositionOperator)
            self.set_rule(('.', Op3), r1, CompositionOperator)
            self.set_rule('.,T', r1, CompositionOperator)
            self.set_rule(('.', Op2), r1, CompositionOperator)
            self.set_rule(('.', OpB), r1, CompositionOperator)
            self.set_rule(('.', Op1), r1, CompositionOperator)
            self.set_rule('.,H', r1, CompositionOperator)
            self.set_rule(('.', Op4), r1, CompositionOperator)
            self.set_rule(('.', Op2), r2, CompositionOperator)

    class Op4(Op3):
        pass

    class OpA(Operator):
        pass

    class OpB(Operator):
        pass

    op = Op3()
    actual = [r.subjects for r in op.rules[CompositionOperator]['left']]
    expected = [
        ('.', 'H'),
        ('.', 'T'),
        ('.', OpB),
        ('.', Op4),
        ('.', Op3),
        ('.', Op2),
        ('.', Op1),
        ('.', OpA),
    ]
    assert actual == expected
    assert op.rules[CompositionOperator]['left'][5].predicate is r2


def assert_merge_identity(op, op1, op2, op_ref):
    assert isinstance(op, type(op_ref))
    attr = {}
    attr.update(op2.attrout)
    attr.update(op1.attrout)
    assert op.attrout == attr
    x = np.ones(op.shapein if op.shapein is not None else 3)
    y = ndarraywrap(4)
    op(x, y)
    if (
        op1.flags.shape_output == 'unconstrained'
        or op2.flags.shape_output == 'unconstrained'
    ):
        y2_tmp = np.empty(3 if isinstance(op2, IdentityOperator) else 4)
        y2 = np.empty(4)
        op2(x, y2_tmp)
        op1(y2_tmp, y2)
    else:
        y2 = op1(op2(x))
    assert_equal(y, y2)
    assert isinstance(y, op1.classout)


@pytest.mark.parametrize('op1', TEST_RULE_OPS)
@pytest.mark.parametrize('op2', IDS_RIGHTS)
def test_merge_identity_right(op1, op2):
    op = op1 @ op2
    assert_merge_identity(op, op1, op2, op1)


@pytest.mark.parametrize('op1', TEST_RULE_OPS)
@pytest.mark.parametrize('op2', IDS_LEFTS)
def test_merge_identity_left(op1, op2):
    op = op2 @ op1
    assert_merge_identity(op, op2, op1, op1)


def test_del_rule():
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self)
            self.set_rule('T', '.')
            self.set_rule('C', '1')
            self.set_rule('.,T', '.', CompositionOperator)
            self.set_rule('T,.', '.', CompositionOperator)
            self.set_rule('.,C', '.I', AdditionOperator)
            self.set_rule('H,.', '.I', AdditionOperator)
            self.set_rule('.,C', '.I', MultiplicationOperator)
            self.set_rule('H,.', '.I', MultiplicationOperator)

    op = Op()
    assert len(op.rules[None]) == 2
    assert len(op.rules[CompositionOperator]['left']) == 1
    assert len(op.rules[CompositionOperator]['right']) == 2
    assert len(op.rules[AdditionOperator]) == 2
    assert len(op.rules[MultiplicationOperator]) == 2

    op.del_rule('T')
    op.del_rule('C')
    op.del_rule('.,T', CompositionOperator)
    op.del_rule('T,.', CompositionOperator)
    op.del_rule('C,.', AdditionOperator)
    op.del_rule('.,H', AdditionOperator)
    op.del_rule('.,C', MultiplicationOperator)
    op.del_rule('H,.', MultiplicationOperator)

    assert len(op.rules[None]) == 0
    assert len(op.rules[CompositionOperator]['left']) == 0
    assert len(op.rules[CompositionOperator]['right']) == 1
    assert len(op.rules[AdditionOperator]) == 0
    assert len(op.rules[MultiplicationOperator]) == 0


@pytest.mark.parametrize(
    'ops, expected_types, expected_values',
    [
        (
            [ho, nl, ho, ar, nl, ho, al, nl, ho],
            [
                HomothetyOperator,
                NonLinearOperator,
                AbsorbRightOperator,
                NonLinearOperator,
                AbsorbLeftOperator,
                NonLinearOperator,
                HomothetyOperator,
            ],
            [2, 0, 6, 0, 6, 0, 2],
        ),
        (
            [ho, nl, ar, ho, nl, al, ho, nl, ho],
            [
                HomothetyOperator,
                NonLinearOperator,
                AbsorbRightOperator,
                NonLinearOperator,
                AbsorbLeftOperator,
                NonLinearOperator,
                HomothetyOperator,
            ],
            [2, 0, 6, 0, 6, 0, 2],
        ),
        (
            [ho, ar, nl, ho, al],
            [AbsorbRightOperator, NonLinearOperator, AbsorbLeftOperator],
            [6, 0, 6],
        ),
        (
            [ar, ho, nl, al, ho],
            [AbsorbRightOperator, NonLinearOperator, AbsorbLeftOperator],
            [6, 0, 6],
        ),
        (
            [ho, ar, li, ho, al],
            [AbsorbRightOperator, LinearOperator, AbsorbLeftOperator],
            [12, 0, 3],
        ),
        (
            [ar, ho, li, al, ho],
            [AbsorbRightOperator, LinearOperator, AbsorbLeftOperator],
            [12, 0, 3],
        ),
        ([ho, li, ar], [LinearOperator, AbsorbRightOperator], [0, 6]),
        ([li, ar, ho], [LinearOperator, AbsorbRightOperator], [0, 6]),
        ([ho, li, al], [LinearOperator, AbsorbLeftOperator], [0, 6]),
        ([li, al, ho], [LinearOperator, AbsorbLeftOperator], [0, 6]),
    ],
)
def test_absorb_scalar(ops, expected_types, expected_values):
    def get_val(op):
        if isinstance(op, (NonLinearOperator, LinearOperator)):
            return 0
        if isinstance(op, HomothetyOperator):
            return op.data
        return op.value

    op = CompositionOperator(ops)
    assert [type(o) for o in op.operands] == expected_types
    assert [get_val(o) for o in op.operands] == expected_values


@pytest.fixture()
def user_rules():
    _old_local_path = pyoperators.config.LOCAL_PATH
    _old_triggers = pyoperators.rules._triggers.copy()
    pyoperators.rules.rule_manager.clear()
    new_local_path = tempfile.gettempdir()
    pyoperators.config.LOCAL_PATH = new_local_path
    with open(os.path.join(new_local_path, 'rules.txt'), 'w') as f:
        f.write(
            """
d1 = 3
d2 = 'value2' # comment
incorrect1

# comment
 # comment
d3 = incorrect2
d4 = 'value4' = incorrect3
d1 = 4"""
        )

    yield

    pyoperators.config.LOCAL_PATH = _old_local_path
    pyoperators.rules._triggers = _old_triggers
    os.remove(os.path.join(tempfile.gettempdir(), 'rules.txt'))


def test_manager(user_rules):
    path = os.path.join(pyoperators.config.LOCAL_PATH, 'rules.txt')
    oldmod = os.stat(path)[0]
    try:
        os.chmod(path, 0)
        with pytest.warns(PyOperatorsWarning):
            RuleManager()
    finally:
        os.chmod(path, oldmod)
    pyoperators.rules.rule_manager.clear()

    with warnings.catch_warnings(record=True) as w:
        rule_manager = RuleManager()
        assert sum(_.category is PyOperatorsWarning for _ in w) == 3
    assert len(rule_manager) == 4
    for key, default in pyoperators.rules._default_triggers.items():
        assert rule_manager[key] == default
    assert 'd1' in rule_manager
    assert rule_manager['d1'] == 4
    assert 'd2' in rule_manager
    assert rule_manager['d2'] == 'value2'
    assert (
        str(rule_manager) == 'd1      = 4         # \n'
        "d2      = 'value2'  # \n"
        'inplace = False     # Allow inplace simplifications\n'
        'none    = False     # Inhibit all rule simplifications'
    )
    rule_manager.register('new_rule', 20, 'my new rule')
    assert 'new_rule' in rule_manager
    assert rule_manager['new_rule'] == 20
    assert pyoperators.rules._description_triggers['new_rule'] == 'my new rule'

    _triggers = pyoperators.rules._triggers
    assert rule_manager.get('d1') == _triggers.get('d1')
    assert rule_manager.items() == _triggers.items()
    assert rule_manager.keys() == _triggers.keys()
    assert rule_manager.pop('d1') == 4
    assert 'd1' not in _triggers
    item = rule_manager.popitem()
    assert item[0] not in _triggers


def test_manager2():
    rule_manager['none'] = False
    assert not rule_manager['none']

    with rule_manager(none=True) as new_rule_manager:
        assert rule_manager['none']

        with new_rule_manager(none=False) as new_rule_manager2:
            assert not rule_manager['none']
            rule_manager['none'] = True
            assert rule_manager['none']
            with new_rule_manager2():
                assert rule_manager['none']

            rule_manager['none'] = False

        assert rule_manager['none']

    assert not rule_manager['none']


def test_manager_errors():
    with pytest.raises(KeyError):
        rule_manager(non_existent_rule=True)
    with pytest.raises(KeyError):
        rule_manager['non_existent']
    with pytest.raises(KeyError):
        rule_manager['non_existent'] = True
    with pytest.raises(TypeError):
        rule_manager.register(32, 0, '')
    with pytest.raises(TypeError):
        rule_manager.register('new_rule', 0, 0)


@pytest.mark.parametrize(
    'cls', [AdditionOperator, CompositionOperator, MultiplicationOperator]
)
@pytest.mark.parametrize('none', [False, True])
def test_rule_manager_none(cls, none):
    op1 = DiagonalOperator([1, 2, 3])
    op2 = 2

    with rule_manager(none=none):
        op = cls([op1, op2])
        if none:
            assert isinstance(op, cls)
        else:
            assert isinstance(op, DiagonalOperator)
