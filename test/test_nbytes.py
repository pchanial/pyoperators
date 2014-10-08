from __future__ import division
import numpy as np
import scipy.sparse as sp
import sys
from numpy.testing import assert_equal
from pyoperators import (
    AdditionOperator, BlockColumnOperator, BlockDiagonalOperator,
    BlockRowOperator, CompositionOperator, MultiplicationOperator,
    Operator, SparseOperator, rule_manager)

COMPOSITES = (AdditionOperator, BlockColumnOperator, BlockDiagonalOperator,
              BlockRowOperator, CompositionOperator, MultiplicationOperator)
PYTHON2 = sys.version_info.major == 2

class Op1(Operator):
    nbytes = 4


class Op2(Operator):
    nbytes = 8


def test_sparse():
    D = np.arange(15, dtype=float).reshape(3, 5)
    matrices = (sp.coo_matrix, sp.bsr_matrix, sp.csc_matrix, sp.csr_matrix,
                sp.dia_matrix, sp.dok_matrix)
    expecteds = 224, 184, 192, 184, 308, 2688 if PYTHON2 else 2464

    def func(matrix, expected):
        op = SparseOperator(matrix(D))
        assert_equal(op.nbytes, expected)
    for matrix, expected in zip(matrices, expecteds):
        yield func, matrix, expected


def test_composite():
    def func(cls):
        if cls in (BlockColumnOperator, BlockDiagonalOperator):
            keywords = {'axisout': 0}
        elif cls is BlockRowOperator:
            keywords = {'axisin': 0}
        else:
            keywords = {}
        op = cls([Op1(), Op2()], **keywords)
        assert_equal(op.nbytes, 12)
        with rule_manager(none=True):
            op = cls([op, Op1(), Op2()], **keywords)
        assert_equal(op.nbytes, 24)
    for cls in COMPOSITES:
        yield func, cls


def test_composite_unique():
    def func(cls):
        if cls in (BlockColumnOperator, BlockDiagonalOperator):
            keywords = {'axisout': 0}
        elif cls is BlockRowOperator:
            keywords = {'axisin': 0}
        else:
            keywords = {}
        op = cls(10 * [Op1(), Op2()], **keywords)
        assert_equal(op.nbytes, 12)
    for cls in COMPOSITES:
        yield func, cls

