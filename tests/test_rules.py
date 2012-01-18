import numpy as np

from numpy.testing import assert_equal
from pyoperators import (Operator, AdditionOperator, CompositionOperator,
         MultiplicationOperator, IdentityOperator, HomothetyOperator,
         ZeroOperator)
from pyoperators.core import OperatorBinaryRule
from pyoperators.utils import assert_is, assert_is_none, assert_is_not_none, assert_is_instance, ndarraywrap

op = Operator()

class ndarray1(np.ndarray):
    pass
class ndarray2(np.ndarray):
    pass
attr1 = { 'attr1': True, 'attr2': True}
attr2 = { 'attr1': False, 'attr3': False}

class ExplExpl(Operator):
    def __init__(self):
        Operator.__init__(self, shapein=3, shapeout=4, classout=ndarray1,
                          attrout=attr1)
    def direct(self, input, output):
        output[0:3] = input
        output[3] = 10.
class UncoExpl(Operator):
    def __init__(self):
        Operator.__init__(self, shapein=3, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:3] = 2*input
        output[3:] = 20
class ImplImpl(Operator):
    def __init__(self):
        Operator.__init__(self, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:input.size] = 3*input
        output[-1] = 30
    def reshapein(self, shapein):
        return (shapein[0] + 1,)
    def reshapeout(self, shapeout):
        return (shapeout[0] - 1,)
class UncoImpl(Operator):
    def __init__(self):
        Operator.__init__(self, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:output.size-1] = 4*input
        output[-1] = 40
    def reshapeout(self, shapeout):
        return (shapeout[0] - 1,)
class ExplUnco(Operator):
    def __init__(self):
        Operator.__init__(self, shapeout=4, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:3] = 5*input[0:3]
        output[3] = 50
class ImplUnco(Operator):
    def __init__(self):
        Operator.__init__(self, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:input.size] = 6*input
        output[-1] = 60
    def reshapein(self, shapein):
        return (shapein[0] + 1,)
class UncoUnco(Operator):
    def __init__(self):
        Operator.__init__(self, classout=ndarray1, attrout=attr1)
    def direct(self, input, output):
        output[0:3] = 7*input[0:3]
        output[3:] = 70
ops = (ExplExpl(), UncoExpl(), ImplImpl(), UncoImpl(), ExplUnco(), ImplUnco(),
       UncoUnco())
ids_right = (IdentityOperator(classout=ndarray2, attrout=attr2),
             IdentityOperator(shapein=3, classout=ndarray2, attrout=attr2))
ids_left = (IdentityOperator(classout=ndarray2, attrout=attr2),
            IdentityOperator(shapein=4, classout=ndarray2, attrout=attr2))
zeros_left = (ZeroOperator(classout=ndarray2, attrout=attr2),
              ZeroOperator(shapein=4, classout=ndarray2, attrout=attr2))

class Operator1(Operator): pass
class Operator2(Operator): pass
class Operator3(Operator): pass
class Operator4(Operator1): pass
op1 = Operator1()
op2 = Operator2()
op3 = Operator3()
op4 = Operator4()

def p1(o1, o2):
    return (o2, op1)
def p2(o1, o2):
    return op3

def test_rule1():
    for s1 in ('.', '.C', '.T', '.H', '.I'):
        o1 = eval('op'+s1) if s1 != '.' else op
        for s2 in ('.', '.C', '.T', '.H', '.I'):
            if '.' not in (s1, s2):
                continue
            o2 = eval('op'+s2) if s2 != '.' else op
            ref = o1 if s2[-1] != '.' else o2
            for s3 in ('1', '.', '.C', '.T', '.H', '.I'):
                rule = OperatorBinaryRule(s1+s2, s3)
                result = rule(o1, o2)
                yield assert_is_not_none, result
                yield assert_is_instance, result, Operator
                if s3 == '1':
                    yield assert_is_instance, result, IdentityOperator
                    continue
                o3 = eval('ref'+s3) if s3 != '.' else ref
                print result, o3
                yield assert_is, result, o3

def test_rule2():
    rule = OperatorBinaryRule('..T', p1)
    yield assert_is_none, rule(op1, op2)
    yield assert_equal, rule(op1, op1.T), (op1.T, op1)

def test_rule3():
    rule = OperatorBinaryRule('..T', p2)
    yield assert_is_none, rule(op1, op2)
    yield assert_is_instance, rule(op1, op1.T), Operator3

def test_rule4():
    rule = OperatorBinaryRule('.{HomothetyOperator}', p1)
    yield assert_is_none, rule(op1, op2)
    s = HomothetyOperator(2)
    yield assert_equal, rule(op1, s), (s, op1)

def test_rule5():
    rule = OperatorBinaryRule('.{self}', p2)
    yield assert_equal, rule(op1, op1), op3
    yield assert_is_none, rule(op1, op2)
    yield assert_equal, rule(op1, op4), op3

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
            func(op, op1, op2, op1)
        for op2 in ids_left:
            op = op2 * op1
            yield func, op, op2, op1, op1

def test_merge_zeros():
    def func(op, op1, op2):
        assert_is_instance(op, ZeroOperator)
        attr = {}
        attr.update(op2.attrout)
        attr.update(op1.attrout)
        assert_equal(op.attrout, attr)
        x = np.ones(3)
        y = ndarraywrap(4)
        op(x, y)
        y2_tmp = np.empty(4)
        y2 = np.empty(4)
        op2(x, y2_tmp)
        op1(y2_tmp, y2)
        assert_equal(y, y2)
        assert_is_instance(y, op1.classout)
    for op1 in zeros_left:
        for op2 in ops:
            op = op1 * op2
            yield func, op, op1, op2

def test_del_rule():
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self)
            self.set_rule('..T', '.', CompositionOperator)
            self.set_rule('.T.', '.', CompositionOperator)
            self.set_rule('..C', 'I', AdditionOperator)
            self.set_rule('.H.', 'I', AdditionOperator)
            self.set_rule('..C', 'I', MultiplicationOperator)
            self.set_rule('.H.', 'I', MultiplicationOperator)

    op = Op()
    nrights = len(op.rules[CompositionOperator]['right'])
    nlefts = len(op.rules[CompositionOperator]['left'])
    assert_equal(len(op.rules[AdditionOperator]), 2)
    assert_equal(len(op.rules[MultiplicationOperator]), 2)

    op.del_rule('..T', CompositionOperator)
    op.del_rule('.T.', CompositionOperator)
    op.del_rule('.C.', AdditionOperator)
    op.del_rule('..H', AdditionOperator)
    op.del_rule('..C', MultiplicationOperator)
    op.del_rule('.H.', MultiplicationOperator)

    yield assert_equal, len(op.rules[CompositionOperator]['right']), nrights-1
    yield assert_equal, len(op.rules[CompositionOperator]['left']), nlefts-1
    yield assert_equal, len(op.rules[AdditionOperator]), 0
    yield assert_equal, len(op.rules[MultiplicationOperator]), 0
