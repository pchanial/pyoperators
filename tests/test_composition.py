from itertools import chain, product, repeat

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import CompositionOperator, Operator, flags
from pyoperators.core import _pool as pool

from .common import OPS, CanUpdateOutput

SHAPES = (None, (), (1,), (3,), (2, 3))


@pytest.mark.parametrize('shapein', SHAPES)
@pytest.mark.parametrize('shapemid', SHAPES)
@pytest.mark.parametrize('shapeout', SHAPES)
def test_composition1(shapein, shapemid, shapeout):
    if shapemid is None and shapein is not None:
        return
    if shapeout is None and shapemid is not None:
        return
    op1 = Operator(shapein=shapein, shapeout=shapemid)
    op2 = Operator(shapein=shapemid, shapeout=shapeout)
    op = op2(op1)
    assert_equal(op.shapein, shapein)
    assert_equal(op.shapeout, shapeout)
    if shapein is not None and shapein == shapeout:
        assert op.flags.square


class OpReshapein(Operator):
    def reshapein(self, shapein):
        return 2 * shapein


@pytest.mark.parametrize('shape', SHAPES)
def test_composition2(shape):
    op = OpReshapein()(Operator(shapeout=shape))
    assert op.shapein is None
    assert op.shapeout == (2 * shape if shape is not None else None)
    assert not op.flags.square


def test_composition3():
    op = OpReshapein()(OpReshapein())
    assert op.shapein is None
    assert op.shapeout is None
    assert not op.flags.square


def test_composition4():
    @flags.linear
    @flags.square
    @flags.inplace
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)

        def direct(self, input, output):
            np.multiply(input, self.v, output)

    pool.clear()
    op = np.product([Op(v) for v in [1]])
    assert op.__class__ is Op
    op(1)
    assert_equal(len(pool), 0)

    pool.clear()
    op = np.product([Op(v) for v in [1, 2]])
    assert op.__class__ is CompositionOperator
    assert_equal(op(1), 2)
    assert_equal(len(pool), 0)

    pool.clear()
    assert_equal(op([1]), 2)
    assert_equal(len(pool), 0)

    op = np.product([Op(v) for v in [1, 2, 4]])
    assert op.__class__ is CompositionOperator

    pool.clear()
    input = np.array(1, int)
    output = np.array(0, int)
    assert_equal(op(input, output), 8)
    assert_equal(input, 1)
    assert_equal(output, 8)
    assert_equal(len(pool), 0)

    pool.clear()
    output = input
    assert_equal(op(input, output), 8)
    assert_equal(input, 8)
    assert_equal(len(pool), 0)

    pool.clear()
    input = np.array([1], int)
    output = np.array([0], int)
    assert_equal(op(input, output), 8)
    assert_equal(input, 1)
    assert_equal(output, 8)
    assert_equal(len(pool), 0)

    pool.clear()
    output = input
    assert_equal(op(input, output), 8)
    assert_equal(input, 8)
    assert_equal(output, 8)
    assert_equal(len(pool), 0)


@pytest.mark.parametrize('flag', ['linear', 'real', 'square', 'separable'])
def test_composition_flags1(flag):
    o = CompositionOperator([Operator(flags=flag), Operator(flags=flag)])
    assert getattr(o.flags, flag)


@pytest.mark.parametrize('flag', ['aligned_input', 'contiguous_input'])
def test_composition_flags2(flag):
    o = CompositionOperator([Operator(), Operator(flags=flag)])
    assert getattr(o.flags, flag)


@pytest.mark.parametrize('flag', ['aligned_output', 'contiguous_output'])
def test_composition_flags3(flag):
    o = CompositionOperator([Operator(flags=flag), Operator()])
    assert getattr(o.flags, flag)


@pytest.mark.parametrize('flag', ['update_output'])
def test_composition_flags4(flag):
    o = CompositionOperator([Operator(), Operator()])
    assert not getattr(o.flags, flag)
    o = CompositionOperator([CanUpdateOutput(), Operator()])
    assert getattr(o.flags, flag)


@pytest.mark.parametrize('cls1', OPS)
@pytest.mark.parametrize('cls2', OPS)
def test_composition_shapes(cls1, cls2):
    n1 = cls1.__name__
    n2 = cls2.__name__
    if n1[4:] == 'Expl' and n2[:4] == 'Expl':
        op = cls1() * cls2(shapeout=3)
    else:
        op = cls1() * cls2()

    shape_output = op.flags.shape_output
    if n1[:4] == 'Unco':
        assert shape_output == 'unconstrained'
    elif n1[:4] == 'Expl':
        assert shape_output == 'explicit'
    elif n2[:4] == 'Expl':
        assert shape_output == 'explicit'
    elif n2[:4] == 'Impl':
        assert shape_output == 'implicit'
    else:
        assert shape_output == 'unconstrained'

    shape_input = op.flags.shape_input
    if n2[4:] == 'Unco':
        assert shape_input == 'unconstrained'
    elif n2[4:] == 'Expl':
        assert shape_input == 'explicit'
    elif n1[4:] == 'Expl':
        assert shape_input == 'explicit'
    elif n1[4:] == 'Impl':
        assert shape_input == 'implicit'
    else:
        assert shape_input == 'unconstrained'


@flags.inplace
class I__(Operator):
    pass


@flags.aligned
@flags.contiguous
class IAC(I__):
    pass


class O____(Operator):
    pass


@flags.aligned_input
@flags.contiguous_input
class O__AC(O____):
    pass


@flags.aligned_output
@flags.contiguous_output
class OAC__(O____):
    pass


@flags.aligned
@flags.contiguous
class OACAC(O____):
    pass


Is = [I__(), IAC()]
Os = [O____(), O__AC(), OAC__(), OACAC()]

TEST_REQS = {
    'I': [[0]],
    'O': [[0], []],
    'II': [[0, 1]],
    'IO': [[0, 1], []],
    'OI': [[0], [1]],
    'OO': [[0], [1], []],
    'III': [[0, 1, 2]],
    'IIO': [[0, 1, 2], []],
    'IOI': [[0, 1], [2]],
    'IOO': [[0, 1], [2], []],
    'OII': [[0], [1, 2]],
    'OIO': [[0], [1, 2], []],
    'OOI': [[0], [1], [2]],
    'OOO': [[0], [1], [2], []],
}


@pytest.mark.parametrize(
    'test, g, ops',
    chain(
        *(
            zip(
                repeat(test),
                repeat(g),
                product(*[Is if _ == 'I' else Os for _ in test]),
            )
            for test, g in TEST_REQS.items()
        )
    ),
)
def test_composition_get_requirements(test, g, ops):
    def get_requirements(ops, test, g):
        rn = [len(_) for _ in g]
        for i in range(len(rn) - 1):
            rn[i] -= 1

        ra = (
            [max(ops[i].flags.aligned_output for i in g[0])]
            + [
                max(
                    [ops[_[0] - 1].flags.aligned_input]
                    + [ops[i].flags.aligned_output for i in _]
                )
                for _ in g[1:-1]
            ]
            + (
                [
                    max(
                        ops[i].flags.aligned_input
                        for i in range(test.rfind('O'), len(ops))
                    )
                ]
                if len(g) > 1
                else []
            )
        )
        rc = (
            [max(ops[i].flags.contiguous_output for i in g[0])]
            + [
                max(
                    [ops[_[0] - 1].flags.contiguous_input]
                    + [ops[i].flags.contiguous_output for i in _]
                )
                for _ in g[1:-1]
            ]
            + (
                [
                    max(
                        ops[i].flags.contiguous_input
                        for i in range(test.rfind('O'), len(ops))
                    )
                ]
                if len(g) > 1
                else []
            )
        )
        return rn, ra, rc

    if len(test) == 1:
        # XXX avoid rule Composition([operator]) -> operator
        # it would be better to disable it.
        c = CompositionOperator(Is)
        c.operands = ops
    else:
        c = CompositionOperator(ops)
    rn1, ra1, rc1 = c._get_requirements()
    rn2, ra2, rc2 = get_requirements(ops, test, g)
    assert rn1 == rn2
    assert ra1 == ra2
    assert rc1 == rc2
