import sys

import numpy as np
import pytest
import scipy.sparse as sp

from pyoperators import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    MultiplicationOperator,
    Operator,
    SparseOperator,
    rule_manager,
)

COMPOSITES = (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    MultiplicationOperator,
)
PYTHON_VERSION = sys.version_info.major == 2


class Op1(Operator):
    nbytes = 4


class Op2(Operator):
    nbytes = 8


@pytest.mark.parametrize(
    'cls, expected',
    [
        (sp.coo_matrix, 224),
        (sp.bsr_matrix, 184),
        (sp.csc_matrix, 192),
        (sp.csr_matrix, 184),
        (sp.dia_matrix, 308),
        (sp.dok_matrix, 2240 if (3, 8) <= sys.version_info < (3, 14) else 2464),
    ],
)
def test_sparse(cls, expected):
    D = np.arange(15, dtype=float).reshape(3, 5)
    op = SparseOperator(cls(D))
    assert op.nbytes == expected


@pytest.mark.parametrize('cls', COMPOSITES)
def test_composite(cls):
    if cls in (BlockColumnOperator, BlockDiagonalOperator):
        keywords = {'axisout': 0}
    elif cls is BlockRowOperator:
        keywords = {'axisin': 0}
    else:
        keywords = {}
    op = cls([Op1(), Op2()], **keywords)
    assert op.nbytes == 12
    with rule_manager(none=True):
        op = cls([op, Op1(), Op2()], **keywords)
    assert op.nbytes == 24


@pytest.mark.parametrize('cls', COMPOSITES)
def test_composite_unique(cls):
    if cls in (BlockColumnOperator, BlockDiagonalOperator):
        keywords = {'axisout': 0}
    elif cls is BlockRowOperator:
        keywords = {'axisin': 0}
    else:
        keywords = {}
    op = cls(10 * [Op1(), Op2()], **keywords)
    assert op.nbytes == 12
