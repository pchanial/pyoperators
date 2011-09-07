import nose
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal

from operators.core import Operator, AdditionOperator, CompositionOperator, ScalarOperator, ndarraywrap
from operators.linear import I, O
from operators.decorators import symmetric, square

dtypes = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64,
          np.float128, np.complex64, np.complex128, np.complex256)]

def assert_flags(operator, flags):
    if isinstance(flags, str):
        flags = flags.split(',')
    for f in flags:
        assert getattr(operator.flags, f.replace(' ', ''))


def test_shapes():
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3,2], np.array([3,2]), (3,2),
                    [3.,2], np.array([3.,2]), (3.,2),
                   ):
        o = Operator(shapein=shapein)
        yield assert_equal, type(o.shapein), tuple
        yield assert_, all([isinstance(s, int) for s in o.shapein])


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
            yield assert_equal, o.dtype, (i * np.array(value, dop)).dtype
            yield assert_array_equal, o, i * np.array(value, dop)


def test_dtype2():
    class Op(Operator):
        def direct(self, input, output):
            np.multiply(input, input, output)
    op = Op()
    input = complex(1,1)
    for di in dtypes:
        try:
            i = np.array(input, di)
        except TypeError:
            i = np.array(input.real, di)
        o = op(i)
        yield assert_equal, o.dtype, (i * i).dtype
        yield assert_array_equal, o, i * i


def test_symmetric():    
    mat = np.matrix([[2,1],[1,2]])
    @symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=(2,), dtype=mat.dtype)
        def direct(self, input, output):
            output[:] = np.dot(mat, input)

    op = Op()
    yield assert_flags, op, 'LINEAR,SQUARE,REAL,SYMMETRIC'
    yield assert_equal, op.shape, (2,2)
    yield assert_equal, op.shapeout, (2,)
    yield assert_, op.C is op.T is op.H is op
    yield assert_, op.I is not op
    yield assert_array_equal, op([1,1]), np.array(mat * [[1],[1]]).ravel()


def test_scalar_reduction1():
    models = 1.+I, -I, (-2) * I, -(2*I), 1.-I, 1.-2*I
    results = [6, -3, -6, -6, 0, -3]
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            o = model(i)
            yield assert_equal, o, result
            yield assert_equal, o.dtype, np.float64

def test_scalar_reduction2():
    model = -I
    iops = '__iadd__', '__isub__', '__imul__', '__iadd__', '__imul__'
    imodels = 2*I, 2*I, 2, O, O
    results = [3, -3, -6, -6, 0]
    for iop, imodel, result in zip(iops, imodels, results):
        model = getattr(model, iop)(imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield assert_equal, model(i), result


def test_decoratein_attribute():
    class ndarray2(np.ndarray):
        pass
    @square
    class AddAttribute(Operator):
        def direct(self, input, output):
            pass
        transpose = direct
        def decoratein(self, input):
            input.newattr_direct = True
        def decorateout(self, input):
            input.newattr_transpose = True
    @square
    class AddAttribute2(Operator):
        def direct(self, input, output):
            pass
        transpose = direct
        def decoratein(self, input):
            input.newattr_direct = False
        def decorateout(self, input):
            input.newattr_transpose = False
    @square
    class AddAttribute3(Operator):
        def direct(self, input, output):
            pass
        transpose = direct
        def decoratein(self, input):
            input.newattr3_direct = True
        def decorateout(self, input):
            input.newattr3_transpose = True

    inputs = [np.ones(5), np.ones(5).view(ndarray2)]
    for i in inputs:
        op = AddAttribute()
        yield assert_, op(i).newattr_direct
        yield assert_, op.T(i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        yield assert_, not op(i).newattr_direct
        yield assert_, op.T(i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        yield assert_, op(i).newattr_direct
        yield assert_, op(i).newattr3_direct
        yield assert_, op.T(i).newattr_transpose
        yield assert_, op.T(i).newattr3_transpose

    for i_ in inputs:
        op = AddAttribute()
        i = i_.copy()
        yield assert_, op(i,i).newattr_direct
        if type(i) is not np.ndarray:
            yield assert_, i.newattr_direct
        i = i_.copy()
        yield assert_, op.T(i,i).newattr_transpose
        if type(i) is not np.ndarray:
            yield assert_, i.newattr_transpose

        op = AddAttribute2() * AddAttribute()
        i = i_.copy()
        yield assert_, not op(i,i).newattr_direct
        if type(i) is not np.ndarray:
            yield assert_, not i.newattr_direct
        i = i_.copy()
        yield assert_, op.T(i,i).newattr_transpose
        if type(i) is not np.ndarray:
            yield assert_, i.newattr_transpose

        op = AddAttribute3() * AddAttribute()
        i = i_.copy()
        o = op(i,i)
        yield assert_, o.newattr_direct
        yield assert_, o.newattr3_direct
        if type(i) is not np.ndarray:
            yield assert_, i.newattr_direct
            yield assert_, i.newattr3_direct
        i = i_.copy()
        o = op.T(i,i)
        yield assert_, o.newattr_transpose
        yield assert_, o.newattr3_transpose
        if type(i) is not np.ndarray:
            assert_, i.newattr_transpose
            assert_, i.newattr3_transpose

    
def test_decoratein_class():
    class ndarray2(np.ndarray):
        pass

    class ndarray3(np.ndarray):
        pass

    class ndarray4(np.ndarray):
        pass

    @square
    class I2(Operator):
        def direct(self, input, output):
            pass
        transpose = direct
        def decoratein(self, output):
            output.newattr = True

    @square
    class I3(Operator):
        def direct(self, input, output):
            pass
        transpose = direct
        def decoratein(self, input):
            input.__class__ = ndarray3
        def decorateout(self, input):
            input.__class__ = ndarray4

    inputs = [np.ones(5), np.ones(5).view(ndarray2)]
    ops = [I, I2(), I2()*I3(), I3()*I2()]
    results = [[np.ndarray, ndarray2],
               [ndarraywrap, ndarray2],
               [ndarray3, ndarray3],
               [ndarray3, ndarray3]]
    resultsT = [[np.ndarray, ndarray2],
                [np.ndarray, ndarray2],
                [ndarray4, ndarray4],
                [ndarray4, ndarray4]]

    for op, results_, resultsT_ in zip(ops, results, resultsT):
        for i, c, cT in zip(inputs, results_, resultsT_):
            o = op(i)
            yield assert_equal, o.__class__, c
            o = op.T(i)
            yield assert_equal, o.__class__, cT

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            i = i.copy()
            if type(i) is np.ndarray:
                op(i,i)
                yield assert_, i.__class__ == np.ndarray
            else:
                op(i,i)
                yield assert_, i.__class__ == c

    for op, resultsT_ in zip(ops, resultsT):
        for i, cT in zip(inputs, resultsT_):
            i = i.copy()
            if type(i) is np.ndarray:
                op.T(i,i)
                yield assert_, i.__class__ == np.ndarray
            else:
                op.T(i,i)
                yield assert_, i.__class__ == cT

def test_addition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.sum([Op(v) for v in [1]])
    yield assert_, op.__class__ is Op

    op = np.sum([Op(v) for v in [1,2]])
    yield assert_, op.__class__ is AdditionOperator
    yield assert_array_equal, op(1), 3
    yield assert_, op.work[0] is not None
    yield assert_, op.work[1] is None

    op = np.sum([Op(v) for v in [1,2,4]])
    yield assert_, op.__class__ is AdditionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    yield assert_array_equal, op(input, output), 7
    yield assert_array_equal, input, 1
    yield assert_array_equal, output, 7
    yield assert_, op.work[0] is not None
    yield assert_, op.work[1] is None

    output = input
    yield assert_array_equal, op(input, output), 7
    yield assert_array_equal, input, 7
    yield assert_array_equal, output, 7
    yield assert_, op.work[0] is not None
    yield assert_, op.work[1] is not None


def test_composition():
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    op = np.product([Op(v) for v in [1]])
    yield assert_, op.__class__ is Op

    op = np.product([Op(v) for v in [1,2]])
    yield assert_, op.__class__ is CompositionOperator
    yield assert_array_equal, op(1), 2
    yield assert_, op.work[0] is None
    yield assert_, op.work[1] is None

    op = np.product([Op(v) for v in [1,2,4]])
    yield assert_, op.__class__ is CompositionOperator

    input = np.array(1, int)
    output = np.array(0, int)
    yield assert_array_equal, op(input, output), 8
    yield assert_array_equal, input, 1
    yield assert_array_equal, output, 8
    yield assert_, op.work[0] is None
    yield assert_, op.work[1] is None

    output = input
    yield assert_array_equal, op(input, output), 8
    yield assert_array_equal, input, 8
    yield assert_array_equal, output, 8
    yield assert_, op.work[0] is None
    yield assert_, op.work[1] is None


def test_scalar_operator():
    s = ScalarOperator(1)
    yield assert_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(-1)
    yield assert_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(2.)
    yield assert_, s.C is s.T is s.H is s
    yield assert_, s.I is not s
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_, isinstance(o, ScalarOperator)

    s = ScalarOperator(complex(0,1))
    yield assert_, s.T is s
    yield assert_, s.H is s.C
    yield assert_, s.I not in (s, s.C)
    yield assert_, s.I.C not in (s, s.C)
    yield assert_, isinstance(s.C, ScalarOperator)
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_, isinstance(o, ScalarOperator)


if __name__ == "__main__":
    nose.run(argv=['', __file__])
