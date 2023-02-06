import itertools

import pytest

from pyoperators import (
    MPI,
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    MultiplicationOperator,
    Operator,
)
from pyoperators.utils import first_is_not

COMPOSITE_CLASSES = [
    AdditionOperator,
    MultiplicationOperator,
    BlockRowOperator,
    BlockDiagonalOperator,
    BlockColumnOperator,
]
ALL_COMMS = [None, MPI.COMM_SELF, MPI.COMM_WORLD]


@pytest.mark.parametrize('cls', COMPOSITE_CLASSES)
@pytest.mark.parametrize('comms', itertools.combinations_with_replacement(ALL_COMMS, 3))
@pytest.mark.parametrize('inout', ['in', 'out'])
def test_comm_composite(cls, comms, inout):
    ops = [Operator(**{'comm' + inout: c}) for c in comms]
    keywords = {}
    args = ()
    if cls in (BlockDiagonalOperator, BlockRowOperator):
        keywords = {'axisin': 0}
    elif cls is BlockColumnOperator:
        keywords = {'axisout': 0}
    else:
        keywords = {}

    if MPI.COMM_SELF in comms and MPI.COMM_WORLD in comms:
        with pytest.raises(ValueError):
            cls(ops, *args, **keywords)
        return
    op = cls(ops, *args, **keywords)
    assert getattr(op, 'comm' + inout) is first_is_not(comms, None)


@pytest.mark.parametrize('commin', ALL_COMMS)
@pytest.mark.parametrize('commout', ALL_COMMS)
def test_comm_composition(commin, commout):
    ops = [Operator(commin=commin), Operator(commout=commout)]
    if None not in (commin, commout) and commin is not commout:
        with pytest.raises(ValueError):
            CompositionOperator(ops)
        return
    op = CompositionOperator(ops)
    assert op.commin is commin
    assert op.commout is commout


COMMIN = MPI.COMM_WORLD.Dup()
COMMOUT = MPI.COMM_WORLD.Dup()


class OpGetComm(Operator):
    def propagate_commin(self, comm):
        return OpNewComm(commin=comm, commout=comm)


class OpNewComm(Operator):
    pass


class OpSetComm1(Operator):
    commin = COMMIN
    commout = COMMOUT


class OpSetComm2(Operator):
    commin = COMMIN
    commout = COMMIN


def test_comm_propagation_composition1():
    op = CompositionOperator([OpGetComm(), OpSetComm1()])
    opget = op.operands[0]
    assert isinstance(opget, OpNewComm)
    assert opget.commin is COMMOUT
    assert opget.commout is COMMOUT


def test_comm_propagation_composition2():
    op = CompositionOperator([OpSetComm1(), OpGetComm()])
    opget = op.operands[1]
    assert isinstance(opget, OpNewComm)
    assert opget.commin is COMMIN
    assert opget.commout is COMMIN


@pytest.mark.parametrize('cls', COMPOSITE_CLASSES)
def test_comm_propagation_composite(cls):
    opgetcomm = OpGetComm()
    opsetcomm2 = OpSetComm2()
    keywords = {}
    if cls in (BlockDiagonalOperator, BlockRowOperator):
        keywords = {'axisin': 0}
    elif cls is BlockColumnOperator:
        keywords = {'axisout': 0}

    for i, ops in enumerate([(opgetcomm, opsetcomm2), (opsetcomm2, opgetcomm)]):
        op = cls(ops, **keywords)

        assert op.commin is COMMIN
        assert op.commout is COMMIN
        opget = op.operands[i]
        assert isinstance(opget, OpNewComm)
        assert opget.commin is COMMIN
        assert opget.commout is COMMIN


@pytest.mark.parametrize('cls', COMPOSITE_CLASSES)
def test_comm_propagation_composition_get_composite_set(cls):
    opgetcomm = OpGetComm()
    opsetcomm2 = OpSetComm2()
    keywords = {}
    if cls in (BlockDiagonalOperator, BlockRowOperator):
        keywords = {'axisin': 0}
    elif cls is BlockColumnOperator:
        keywords = {'axisout': 0}

    for i, ops in enumerate(
        [(opgetcomm(Operator()), opsetcomm2), (opsetcomm2, Operator()(opgetcomm))]
    ):
        op = cls(ops, **keywords)

        assert op.commin is COMMIN
        assert op.commout is COMMIN
        compget = op.operands[i]
        assert compget.commin is COMMIN
        assert compget.commout is COMMIN
        opget = op.operands[i].operands[i]
        assert isinstance(opget, OpNewComm)
        assert opget.commin is COMMIN
        assert opget.commout is COMMIN


@pytest.mark.parametrize('cls', COMPOSITE_CLASSES)
def test_comm_propagation_composite_set_composition_get(cls):
    opgetcomm = OpGetComm()
    opsetcomm2 = OpSetComm2()
    keywords = {}
    if cls in (BlockDiagonalOperator, BlockRowOperator):
        keywords = {'axisin': 0}
    elif cls is BlockColumnOperator:
        keywords = {'axisout': 0}

    for ops_in in [(opsetcomm2, Operator()), (Operator(), opsetcomm2)]:
        op_in = cls(ops_in, **keywords)
        for i, op in enumerate([opgetcomm(op_in), op_in(opgetcomm)]):
            assert op.commin is COMMIN
            assert op.commout is COMMIN
            opget = op.operands[i]
            assert isinstance(opget, OpNewComm)
            assert opget.commin is COMMIN
            assert opget.commout is COMMIN


@pytest.mark.parametrize('cls', COMPOSITE_CLASSES)
def test_comm_propagation_composite_get_composition_set(cls):
    opgetcomm = OpGetComm()
    opsetcomm2 = OpSetComm2()
    keywords = {}
    if cls in (BlockDiagonalOperator, BlockRowOperator):
        keywords = {'axisin': 0}
    elif cls is BlockColumnOperator:
        keywords = {'axisout': 0}

    for i, ops_in in enumerate([(opgetcomm, Operator()), (Operator(), opgetcomm)]):
        op_in = cls(ops_in, **keywords)
        for j, op in enumerate([op_in(opsetcomm2), opsetcomm2(op_in)]):
            assert op.commin is COMMIN
            assert op.commout is COMMIN
            compget = op.operands[j]
            assert compget.commin is COMMIN
            assert compget.commout is COMMIN
            opget = compget.operands[i]
            assert isinstance(opget, OpNewComm)
            assert opget.commin is COMMIN
            assert opget.commout is COMMIN
