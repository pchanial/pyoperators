import nose
import numpy as np
from numpy.testing import *

from operators import Symmetric, Operator, AdditionOperator, CompositionOperator, ScalarOperator, I, O

dtypes = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64,
          np.float128, np.complex64, np.complex128, np.complex256)]

def assert_flags(operator, flags):
    if isinstance(flags, str):
        flags = flags.split(',')
    for f in flags:
        assert getattr(operator.flags, f.replace(' ', ''))


def test_dtype1():

    value = 2.5
    class Op(Operator):
        def __init__(self, dtype):
            Operator.__init__(self, dtype=dtype)
        def direct(self, input, output):
            np.multiply(input, np.array(value, self.dtype), output)

    input = complex(1,1)
    for dop in dtypes:
        for di in dtypes:
            try:
                i = np.array(input, di)
            except TypeError:
                i = np.array(input.real, di)
            o = Op(dop)(i)
            assert o.dtype == (i * np.array(value, dop)).dtype
            assert_array_equal(o, i * np.array(value, dop))


def test_dtype2():
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)

    input = complex(1,1)
    for di in dtypes:
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = Op()(i)
        print o.dtype, o
        assert o.dtype == (i * i).dtype
        assert_array_equal(o, i * i)


def test_symmetric():
    
    mat = np.matrix([[2,1],[1,2]])
    @Symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=(2,), dtype=mat.dtype)
        def direct(self, input, output):
            output[:] = np.dot(mat, input)

    op = Op()
    assert_flags(op, 'LINEAR,SQUARE,REAL,SYMMETRIC')
    assert op.shape == (2,2)
    assert op.shapeout == (2,)
    assert op.C is op.T is op.H is op
    assert op.I is not op
    assert_array_equal(op([1,1]), np.array(mat * [[1],[1]]).ravel())


def test_scalar_reduction():

    models = 1.+I, -I, (-2) * I, -(2*I), 1.-I, 1.-2*I
    results = [6, -3, -6, -6, 0, -3]
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            assert model(i) == result
            assert model(i).dtype == np.float64

    iops = '__iadd__', '__isub__', '__imul__', '__iadd__', '__imul__'
    imodels = 2*I, 2*I, 2, O, O
    results = [3, -3, -6, -6, 0]
    for iop, imodel, result in zip(iops, imodels, results):
        model = getattr(model, iop)(imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            assert model(i) == result
            assert model(i).dtype == np.float64


def test_addition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.sum([Op(v) for v in [1]])
    assert op.__class__ is Op

    op = np.sum([Op(v) for v in [1,2]])
    assert op.__class__ is AdditionOperator
    assert_array_equal(op(1), 3)
    assert op.work[0] is not None
    assert op.work[1] is None

    op = np.sum([Op(v) for v in [1,2,4]])
    assert op.__class__ is AdditionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    assert_array_equal(op(input, output), 7)
    assert_array_equal(input, 1)
    assert_array_equal(output, 7)
    assert op.work[0] is not None
    assert op.work[1] is None

    output = input
    assert_array_equal(op(input, output), 7)
    assert_array_equal(input, 7)
    assert_array_equal(output, 7)
    assert op.work[0] is not None
    assert op.work[1] is not None


def test_composition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.product([Op(v) for v in [1]])
    assert op.__class__ is Op

    op = np.product([Op(v) for v in [1,2]])
    assert op.__class__ is CompositionOperator
    assert_array_equal(op(1), 2)
    assert op.work[0] is None
    assert op.work[1] is None

    op = np.product([Op(v) for v in [1,2,4]])
    assert op.__class__ is CompositionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 1)
    assert_array_equal(output, 8)
    assert op.work[0] is None
    assert op.work[1] is None

    output = input
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 8)
    assert_array_equal(output, 8)
    assert op.work[0] is None
    assert op.work[1] is None


def test_scalar_operator():
    s = ScalarOperator(1)
    assert s.C is s.T is s.H is s.I is s

    s = ScalarOperator(-1)
    assert s.C is s.T is s.H is s.I is s

    s = ScalarOperator(2.)
    assert s.C is s.T is s.H is s
    assert s.I is not s
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        assert isinstance(o, ScalarOperator)

    s = ScalarOperator(complex(0,1))
    assert s.T is s
    assert s.H is s.C
    assert s.I not in (s, s.C)
    assert s.I.C not in (s, s.C)
    assert isinstance(s.C, ScalarOperator)
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        assert isinstance(o, ScalarOperator)


if __name__ == "__main__":
    nose.run(argv=['', __file__])
