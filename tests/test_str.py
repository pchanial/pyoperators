import pytest

from pyoperators import (
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    I,
    Operator,
    PowerOperator,
    ProductOperator,
    flags,
)


class NL1(Operator):
    def __str__(self):
        return 'n'


class NL2(Operator):
    def __str__(self):
        return 'a(..., z=1)'


class NL3(Operator):
    def __str__(self):
        return '...**2'


@flags.linear
class L(Operator):
    pass


n = NL1()
l = L()
a = NL2()
b = NL3()
br = BlockRowOperator([I, 2 * I], axisin=0, partitionin=(3, 3))
bd = BlockDiagonalOperator([I, 2 * I], axisin=0, partitionin=(3, 3))
bc = BlockColumnOperator([I, 2 * I], axisout=0, partitionout=(3, 3))
add = l + n
mul = l * n


@pytest.mark.parametrize(
    'operands, expected_str',
    [
        ([n], 'n'),
        ([l], 'l'),
        ([n, n], 'n @ n'),
        ([n, l], 'n @ l'),
        ([l, n], 'l @ n'),
        ([l, l], 'l @ l'),
        ([n, n, n], 'n @ n @ n'),
        ([n, n, l], 'n @ n @ l'),
        ([n, l, n], 'n @ l @ n'),
        ([n, l, l], 'n @ l @ l'),
        ([l, n, n], 'l @ n @ n'),
        ([l, n, l], 'l @ n @ l'),
        ([l, l, n], 'l @ l @ n'),
        ([l, l, l], 'l @ l @ l'),
    ],
)
def test1(operands, expected_str):
    assert str(CompositionOperator(operands)) == expected_str


@pytest.mark.parametrize(
    'operands, expected_str',
    [
        ([a], 'a(..., z=1)'),
        ([n, a], 'n @ a(..., z=1)'),
        ([a, n], 'a(n, z=1)'),
        ([a, a], 'a(a(..., z=1), z=1)'),
        ([n, n, a], 'n @ n @ a(..., z=1)'),
        ([n, a, n], 'n @ a(n, z=1)'),
        ([n, a, a], 'n @ a(a(..., z=1), z=1)'),
        ([a, n, n], 'a(n @ n, z=1)'),
        ([a, n, a], 'a(n @ a(..., z=1), z=1)'),
        ([a, a, n], 'a(a(n, z=1), z=1)'),
        ([a, a, a], 'a(a(a(..., z=1), z=1), z=1)'),
    ],
)
def test2(operands, expected_str):
    assert str(CompositionOperator(operands)) == expected_str


@pytest.mark.parametrize(
    'operands, expected_str',
    [
        ([b], '...**2'),
        ([n, b], 'n @ (...**2)'),
        ([b, n], 'n**2'),
        ([b, b], '(...**2)**2'),
        ([n, n, b], 'n @ n @ (...**2)'),
        ([n, b, n], 'n @ n**2'),
        ([n, b, b], 'n @ (...**2)**2'),
        ([b, n, n], '(n @ n)**2'),
        ([b, n, b], '(n @ (...**2))**2'),
        ([b, b, n], '(n**2)**2'),
        ([b, b, b], '((...**2)**2)**2'),
    ],
)
def test3(operands, expected_str):
    assert str(CompositionOperator(operands)) == expected_str


@pytest.mark.parametrize(
    'operator, expected_str',
    [
        (PowerOperator(3) @ ProductOperator(axis=2), 'product(..., axis=2)**3'),
        (bc, '[ [I] [2I] ]'),
        (bd, 'I ⊕ 2I'),
        (br, '[[ I 2I ]]'),
        (add, 'l + n'),
        (mul, 'l × n'),
        (l @ bc, 'l @ [ [I] [2I] ]'),
        (l @ bd, 'l @ (I ⊕ 2I)'),
        (l @ br, 'l @ [[ I 2I ]]'),
        (l @ add, 'l @ (l + n)'),
        (l @ mul, 'l @ (l × n)'),
        (bc @ l, '[ [I] [2I] ] @ l'),
        (bd @ l, '(I ⊕ 2I) @ l'),
        (br @ l, '[[ I 2I ]] @ l'),
        (add @ l, '(l + n) @ l'),
        (mul @ l, '(l × n) @ l'),
        (l @ bc @ l, 'l @ [ [I] [2I] ] @ l'),
        (l @ bd @ l, 'l @ (I ⊕ 2I) @ l'),
        (l @ br @ l, 'l @ [[ I 2I ]] @ l'),
        (l @ add @ l, 'l @ (l + n) @ l'),
        (l @ mul @ l, 'l @ (l × n) @ l'),
    ],
)
def test4(operator, expected_str):
    assert str(operator) == expected_str
