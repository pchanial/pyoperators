import nose
from nose.tools import eq_

from pyoperators import Operator, IdentityOperator, ScalarOperator
from pyoperators.core import OperatorRule
from pyoperators.utils import assert_is, assert_is_none, assert_is_not_none, assert_is_instance

op = Operator()

class Operator1(Operator): pass
class Operator2(Operator): pass
class Operator3(Operator): pass
class Operator4(Operator1): pass
op1 = Operator1()
op2 = Operator2()
op3 = Operator3()
op4 = Operator4()

def p1(o2):
    return (o2, op1)
def p2(o2):
    return op3

def test_rule1():
    for s1 in ('.', '.C', '.T', '.H', '.I'):
        o1 = eval('op'+s1) if s1 != '.' else op
        for s2 in ('.', '.C', '.T', '.H', '.I'):
            if '.' not in (s1, s2):
                continue
            o2 = eval('op'+s2) if s2 != '.' else op
            ref, other = (o1, o2) if s2[-1] != '.' else (o2, o1)
            for s3 in ('1', '.', '.C', '.T', '.H', '.I'):
                print s1, s2, s3
                rule = OperatorRule(ref, s1+s2, s3)
                result = rule(other)
                yield assert_is_not_none, result
                yield assert_is_instance, result, Operator
                if s3 == '1':
                    yield assert_is_instance, result, IdentityOperator
                    continue
                o3 = eval('ref'+s3) if s3 != '.' else ref
                print result, o3
                yield assert_is, result, o3

def test_rule2():
    rule = OperatorRule(op1, '..T', p1)
    yield assert_is_none, rule(op2)
    yield eq_, rule(op1.T), (op1.T, op1)

def test_rule3():
    rule = OperatorRule(op1, '..T', p2)
    yield assert_is_none, rule(op2)
    yield assert_is_instance, rule(op1.T), Operator3

def test_rule4():
    rule = OperatorRule(op1, '.{ScalarOperator}', p1)
    yield assert_is_none, rule(op2)
    s = ScalarOperator(2)
    yield eq_, rule(s), (s, op1)

def test_rule5():
    rule = OperatorRule(op1, '.{self}', p2)
    yield eq_, rule(op1), op3
    yield assert_is_none, rule(op2)
    yield eq_, rule(op4), op3

if __name__ == "__main__":
    nose.run(argv=['', __file__])
