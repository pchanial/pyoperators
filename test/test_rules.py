import numpy as np
import os
import pyoperators
import tempfile
import warnings
from nose import with_setup
from numpy.testing import assert_equal, assert_raises, assert_warns
from pyoperators import (
    Operator, AdditionOperator, CompositionOperator, DiagonalOperator,
    HomothetyOperator, IdentityOperator, MultiplicationOperator,
    PyOperatorsWarning)
from pyoperators.flags import linear
from pyoperators.rules import BinaryRule, UnaryRule, RuleManager, rule_manager
from pyoperators.utils import ndarraywrap
from pyoperators.utils.testing import (
    assert_eq, assert_is, assert_is_none, assert_is_not_none,
    assert_is_instance)

from .common import OPS, ndarray2, attr2

op = Operator()
ops = [OP() for OP in OPS]

ids_left = (IdentityOperator(classout=ndarray2, attrout=attr2),
            IdentityOperator(shapein=4, classout=ndarray2, attrout=attr2))
ids_right = (IdentityOperator(classout=ndarray2, attrout=attr2),
             IdentityOperator(shapein=3, classout=ndarray2, attrout=attr2))


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


def p1(o1, o2):
    return (o2, op1)


def p2(o1, o2):
    return op3


def test_unaryrule1():
    def func(s, p):
        r = UnaryRule(s, p)
        if p == '.':
            assert_is(r(op1), op1)
        else:
            assert_is_instance(r(op1), IdentityOperator)
    for s in ('C', 'T', 'H', 'I', 'IC', 'IT', 'IH'):
        for p in ('.', '1'):
            yield func, s, p


def test_unaryrule2():
    assert_raises(ValueError, UnaryRule, '.', '.')
    assert_raises(ValueError, UnaryRule, '.T', '.')
    assert_raises(ValueError, UnaryRule, 'T', 'C')
    assert_raises(ValueError, UnaryRule, 'T', 'T')
    assert_raises(ValueError, UnaryRule, 'T', 'H')
    assert_raises(ValueError, UnaryRule, 'T', 'I')


def test_binaryrule1():
    op.T  # generate associated operators

    def func(s1, s2, s3, o1, o2, ref):
            rule = BinaryRule(s1 + ',' + s2, s3)
            result = rule(o1, o2)
            assert_is_not_none(result)
            assert_is_instance(result, Operator)
            if s3 == '1':
                assert_is_instance(result, IdentityOperator)
                return
            o3 = eval('ref.'+s3) if s3 != '.' else ref
            assert_is(result, o3)
    for s1 in ('.', 'C', 'T', 'H', 'I'):
        o1 = eval('op.'+s1) if s1 != '.' else op
        for s2 in ('.', 'C', 'T', 'H', 'I'):
            if '.' not in (s1, s2):
                continue
            o2 = eval('op.'+s2) if s2 != '.' else op
            ref = o1 if s2[-1] != '.' else o2
            for s3 in ('1', '.', 'C', 'T', 'H', 'I'):
                yield func, s1, s2, s3, o1, o2, ref


def test_binaryrule2():
    rule = BinaryRule('.,T', p1)
    yield assert_is_none, rule(op1, op2)
    yield assert_equal, rule(op1, op1.T), (op1.T, op1)


def test_binaryrule3():
    rule = BinaryRule('.,T', p2)
    yield assert_is_none, rule(op1, op2)
    yield assert_is_instance, rule(op1, op1.T), Operator3


def test_binaryrule4():
    rule = BinaryRule(('.', HomothetyOperator), p1)
    yield assert_is_none, rule(op1, op2)
    s = HomothetyOperator(2)
    yield assert_equal, rule(op1, s), (s, op1)


def test_binaryrule5():
    rule = BinaryRule((type(op1), '.'), p2)
    yield assert_equal, rule(op1, op1), op3
    yield assert_is_none, rule(op2, op1)
    yield assert_equal, rule(op4, op1), op3


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
    act = [r.subjects for r in op.rules[CompositionOperator]['left']]
    exp = [('.', 'H'), ('.', 'T'), ('.', OpB), ('.', Op4), ('.', Op3),
           ('.', Op2), ('.', Op1), ('.', OpA)]
    for a, e in zip(act, exp):
        yield assert_eq, a, e
    assert op.rules[CompositionOperator]['left'][5].predicate is r2


def test_merge_identity():
    def func(op, op1, op2, op_ref):
        assert_is_instance(op, type(op_ref))
        attr = {}
        attr.update(op2.attrout)
        attr.update(op1.attrout)
        assert_equal(op.attrout, attr)
        x = np.ones(op.shapein if op.shapein is not None else 3)
        y = ndarraywrap(4)
        op(x, y)
        if op1.flags.shape_output == 'unconstrained' or \
           op2.flags.shape_output == 'unconstrained':
            y2_tmp = np.empty(3 if isinstance(op2, IdentityOperator) else 4)
            y2 = np.empty(4)
            op2(x, y2_tmp)
            op1(y2_tmp, y2)
        else:
            y2 = op1(op2(x))
        assert_equal(y, y2)
        assert_is_instance(y, op1.classout)
    for op1 in ops:
        for op2 in ids_right:
            op = op1 * op2
            yield func, op, op1, op2, op1
        for op2 in ids_left:
            op = op2 * op1
            yield func, op, op2, op1, op1


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
    assert_equal(len(op.rules[None]), 2)
    assert_equal(len(op.rules[CompositionOperator]['left']), 1)
    assert_equal(len(op.rules[CompositionOperator]['right']), 2)
    assert_equal(len(op.rules[AdditionOperator]), 2)
    assert_equal(len(op.rules[MultiplicationOperator]), 2)

    op.del_rule('T')
    op.del_rule('C')
    op.del_rule('.,T', CompositionOperator)
    op.del_rule('T,.', CompositionOperator)
    op.del_rule('C,.', AdditionOperator)
    op.del_rule('.,H', AdditionOperator)
    op.del_rule('.,C', MultiplicationOperator)
    op.del_rule('H,.', MultiplicationOperator)

    assert_equal(len(op.rules[None]), 0)
    assert_equal(len(op.rules[CompositionOperator]['left']), 0)
    assert_equal(len(op.rules[CompositionOperator]['right']), 1)
    assert_equal(len(op.rules[AdditionOperator]), 0)
    assert_equal(len(op.rules[MultiplicationOperator]), 0)


def test_absorb_scalar():
    h = HomothetyOperator(2)

    @linear
    class AbsorbRightOperator(Operator):
        def __init__(self, value=3., **keywords):
            self.value = np.array(value)
            Operator.__init__(self, **keywords)
            self.set_rule(('.', HomothetyOperator), lambda s, o:
                          AbsorbRightOperator(s.value * o.data),
                          CompositionOperator)

    @linear
    class AbsorbLeftOperator(Operator):
        def __init__(self, value=3., **keywords):
            self.value = np.array(value)
            Operator.__init__(self, **keywords)
            self.set_rule((HomothetyOperator, '.'), lambda o, s:
                          AbsorbLeftOperator(s.value * o.data),
                          CompositionOperator)
    nl = NonLinearOperator()
    l = LinearOperator()
    ar = AbsorbRightOperator()
    al = AbsorbLeftOperator()
    ops = [[h, nl, h, ar, nl, h, al, nl, h],
           [h, nl, ar, h, nl, al, h, nl, h],
           [h, ar, nl, h, al],
           [ar, h, nl, al, h],
           [h, ar, l, h, al],
           [ar, h, l, al, h],
           [h, l, ar],
           [l, ar, h],
           [h, l, al],
           [l, al, h]]
    expected_types = [
        [HomothetyOperator, NonLinearOperator, AbsorbRightOperator,
         NonLinearOperator, AbsorbLeftOperator, NonLinearOperator,
         HomothetyOperator],
        [HomothetyOperator, NonLinearOperator, AbsorbRightOperator,
         NonLinearOperator, AbsorbLeftOperator, NonLinearOperator,
         HomothetyOperator],
        [AbsorbRightOperator, NonLinearOperator, AbsorbLeftOperator],
        [AbsorbRightOperator, NonLinearOperator, AbsorbLeftOperator],
        [AbsorbRightOperator, LinearOperator, AbsorbLeftOperator],
        [AbsorbRightOperator, LinearOperator, AbsorbLeftOperator],
        [LinearOperator, AbsorbRightOperator],
        [LinearOperator, AbsorbRightOperator],
        [LinearOperator, AbsorbLeftOperator],
        [LinearOperator, AbsorbLeftOperator]]
    expected_values = [[2, 0, 6, 0, 6, 0, 2],
                       [2, 0, 6, 0, 6, 0, 2],
                       [6, 0, 6],
                       [6, 0, 6],
                       [12, 0, 3],
                       [12, 0, 3],
                       [0, 6],
                       [0, 6],
                       [0, 6],
                       [0, 6]]

    def get_val(op):
        if isinstance(op, (NonLinearOperator, LinearOperator)):
            return 0
        if isinstance(op, HomothetyOperator):
            return op.data
        return op.value

    def func(ops, expected_types, expected_values):
        op = CompositionOperator(ops)
        assert_eq([type(o) for o in op.operands], expected_types)
        assert_eq([get_val(o) for o in op.operands], expected_values)
    for op, expected_type, expected_value in zip(ops, expected_types,
                                                 expected_values):
        yield func, op, expected_type, expected_value


_old_local_path = None
_old_triggers = None


def setup_user_rules():
    global _old_local_path, _old_triggers
    _old_local_path = pyoperators.config.LOCAL_PATH
    _old_triggers = pyoperators.rules._triggers.copy()
    pyoperators.rules.rule_manager.clear()
    new_local_path = tempfile.gettempdir()
    pyoperators.config.LOCAL_PATH = new_local_path
    with open(os.path.join(new_local_path, 'rules.txt'), 'w') as f:
        f.write("""
d1 = 3
d2 = 'value2' # comment
incorrect1

# comment
 # comment
d3 = incorrect2
d4 = 'value4' = incorrect3
d1 = 4""")


def teardown_user_rules():
    pyoperators.config.LOCAL_PATH = _old_local_path
    pyoperators.rules._triggers = _old_triggers
    os.remove(os.path.join(tempfile.gettempdir(), 'rules.txt'))


@with_setup(setup_user_rules, teardown_user_rules)
def test_manager():
    path = os.path.join(pyoperators.config.LOCAL_PATH, 'rules.txt')
    oldmod = os.stat(path)[0]
    try:
        os.chmod(path, 0)
        assert_warns(PyOperatorsWarning, RuleManager)
    finally:
        os.chmod(path, oldmod)
    pyoperators.rules.rule_manager.clear()

    with warnings.catch_warnings(record=True) as w:
        rule_manager = RuleManager()
        assert_equal(sum(_.category is PyOperatorsWarning for _ in w), 3)
    assert_equal(len(rule_manager), 4)
    for key, default in pyoperators.rules._default_triggers.items():
        assert_equal(rule_manager[key], default)
    assert 'd1' in rule_manager
    assert_equal(rule_manager['d1'], 4)
    assert 'd2' in rule_manager
    assert_equal(rule_manager['d2'], 'value2')
    assert_equal(str(rule_manager),
                 'd1      = 4         # \n'
                 "d2      = 'value2'  # \n"
                 'inplace = False     # Allow inplace simplifications\n'
                 'none    = False     # Inhibit all rule simplifications')
    rule_manager.register('new_rule', 20, 'my new rule')
    assert 'new_rule' in rule_manager
    assert_equal(rule_manager['new_rule'], 20)
    assert_equal(pyoperators.rules._description_triggers['new_rule'],
                 'my new rule')

    _triggers = pyoperators.rules._triggers
    assert_equal(rule_manager.get('d1'), _triggers.get('d1'))
    assert_equal(rule_manager.items(), _triggers.items())
    assert_equal(rule_manager.keys(), _triggers.keys())
    assert_equal(rule_manager.pop('d1'), 4)
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
    assert_raises(KeyError, rule_manager, non_existent_rule=True)
    assert_raises(KeyError, rule_manager.__getitem__, 'non_existent')
    assert_raises(KeyError, rule_manager.__setitem__, 'non_existent', True)
    assert_raises(TypeError, rule_manager.register, 32, 0, '')
    assert_raises(TypeError, rule_manager.register, 'new_rule', 0, 0)


def test_rule_manager_none():
    op1 = DiagonalOperator([1, 2, 3])
    op2 = 2

    def func(cls, none):
        with rule_manager(none=none):
            op = cls([op1, op2])
            if none:
                assert_is_instance(op, cls)
            else:
                assert_is_instance(op, DiagonalOperator)
    for cls in AdditionOperator, CompositionOperator, MultiplicationOperator:
        for none in False, True:
            yield func, cls, none
