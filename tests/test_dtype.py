import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import Operator, flags

from .common import DTYPES


@pytest.mark.parametrize('dop', DTYPES)
@pytest.mark.parametrize('di', DTYPES)
def test_dtype1(dop, di):
    @flags.square
    class Op(Operator):
        def __init__(self, dtype):
            Operator.__init__(self, dtype=dtype)

        def direct(self, input, output):
            np.multiply(input, np.array(value, self.dtype), output)

    value = 2.5
    input = complex(1, 1)
    try:
        i = np.array(input, di)
    except TypeError:
        i = np.array(input.real, di)
    o = Op(dop)(i)
    assert_equal(o.dtype, (i * np.array(value, dop)).dtype, str((dop, di)))
    assert_equal(o, i * np.array(value, dop), str((dop, di)))


@pytest.mark.parametrize('di', DTYPES)
def test_dtype2(di):
    @flags.linear
    @flags.square
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)

    op = Op()
    input = complex(1, 1)

    try:
        i = np.array(input, di)
    except TypeError:
        i = np.array(input.real, di)
    o = op(i)
    assert_equal(o.dtype, (i * i).dtype, str(di))
    assert_equal(o, i * i, str(di))


@pytest.mark.parametrize('dtypein', DTYPES)
@pytest.mark.parametrize('dtypeout', DTYPES)
def test_dtypein_dtypeout(dtypein, dtypeout):
    @flags.square
    class MyOp(Operator):
        def direct(self, input, output):
            assert input.dtype == dtypein
            assert output.dtype == dtypeout
            output[...] = input

    op = MyOp(dtypein=dtypein, dtypeout=dtypeout)
    assert op(1).dtype == dtypeout
