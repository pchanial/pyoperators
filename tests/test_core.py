import nose
from nose.tools import (eq_, ok_, assert_is, assert_is_not, assert_not_in,
                        assert_is_instance)
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_raises

from pyoperators import memory
from pyoperators.core import (Operator, AdditionOperator, CompositionOperator,
         PartitionOperator, ExpansionOperator, ReductionOperator,
         ScalarOperator, asoperator)
from pyoperators.decorators import symmetric, square, inplace
from pyoperators.linear import I, O, DiagonalOperator, IdentityOperator
from pyoperators.utils import ndarraywrap

np.seterr(all='raise')

memory.verbose = True

def assert_flags(operator, flags, msg=''):
    if isinstance(flags, str):
        flags = [f.replace(' ', '') for f in flags.split(',')]
    for f in flags:
        assert getattr(operator.flags, f), 'Operator {0} is not {1}.'.format(
            operator, f) + (' ' + msg if msg else '')

def assert_flags_false(operator, flags, msg=''):
    if isinstance(flags, str):
        flags = [f.replace(' ', '') for f in flags.split(',')]
    for f in flags:
        assert not getattr(operator.flags, f),'Operator {0} is not {1}.'.format(
            operator, f) + (' ' + msg if msg else '')

def assert_is_inttuple(shape, msg=''):
    msg = '{0} is not an int tuple.'.format(shape) + (' ' + msg if msg else '')
    assert type(shape) is tuple, msg
    assert all([isinstance(s, int) for s in shape]), msg

dtypes = [np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
          np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64,
          np.float128, np.complex64, np.complex128, np.complex256)]

class ndarray2(np.ndarray):
    pass

class ndarray3(np.ndarray):
    pass

class ndarray4(np.ndarray):
    pass

@square
class Op2(Operator):
    attrout = {'newattr':True}
    def direct(self, input, output):
        pass
    def transpose(self, input, output):
        pass            

@square
class Op3(Operator):
    classout = ndarray3
    classin = ndarray4
    def direct(self, input, output):
        pass
    def transpose(self, input, output):
        pass

def test_shape_is_inttuple():
    for shapein in (3, [3], np.array(3), np.array([3]), (3,),
                    3., [3.], np.array(3.), np.array([3.]), (3.,),
                    [3,2], np.array([3,2]), (3,2),
                    [3.,2], np.array([3.,2]), (3.,2)):
        o = Operator(shapein=shapein, shapeout=shapein)
        yield assert_is_inttuple, o.shapein, 'shapein: '+str(shapein)
        yield assert_is_inttuple, o.shapeout, 'shapeout: '+str(shapein)

def test_shape_explicit():
    o1, o2, o3 = (
        Operator(shapeout=(13,2), shapein=(2,2)),
        Operator(shapeout=(2,2), shapein=(1,3)),
        Operator(shapeout=(1,3), shapein=4))
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((13,2),(2,2),(13,2)),
                            ((1,3),(4,),(4,))):
        yield assert_equal, o.shapeout, eout, '*shapeout:'+str(o)
        yield assert_equal, o.shapein, ein, '*shapein:'+str(o)
    yield assert_raises, ValueError, CompositionOperator, [o2, o1]
    yield assert_raises, ValueError, CompositionOperator, [o3, o2]
    yield assert_raises, ValueError, CompositionOperator, [o3, I, o1]

    o4 = Operator(shapeout=o1.shapeout)
    o5 = Operator(flags={'SQUARE':True})

    o1 = Operator(shapein=(13,2))
    for o in [o1+I, I+o1, o1+o4, o1+I+o5+o4, I+o5+o1]:
        yield assert_equal, o.shapein, o1.shapein, '+shapein:'+str(o)
        yield assert_equal, o.shapeout, o1.shapeout, '+shapeout:'+str(o)
    yield assert_raises, ValueError, AdditionOperator, [o2, o1]
    yield assert_raises, ValueError, AdditionOperator, [o3, o2]
    yield assert_raises, ValueError, AdditionOperator, [I, o3, o1]
    yield assert_raises, ValueError, AdditionOperator, [o3, I, o1]

def test_shape_implicit():
    class Op(Operator):
        def __init__(self, factor):
            self.factor = factor
            Operator.__init__(self)
        def reshapein(self, shape):
            if shape is None: return None
            return shape[0]*self.factor
        def reshapeout(self, shape):
            if shape is None: return None
            return shape[0]/self.factor
        def __str__(self):
            return super(Op, self).__str__() + ' {0}'.format(self.factor)
    o1, o2, o3 = (Op(2), Op(3), Op(4))
    assert o1.shapein is o2.shapein is o3.shapein is None
    shapein = (1,)
    shapeout = (24,)
    for o, eout, ein in zip([o1*o2, o2*o3, o1*o2*o3],
                            ((6,),(12,),(24,)),
                            ((4,),(2,),(1,))):
        yield assert_equal, o.reshapein(shapein), eout, 'reshapein:'+str(o)
        yield assert_equal, o.reshapeout(shapeout), ein, 'reshapeout:'+str(o)

def test_shape_cornercases():
    op = Operator()
    assert_flags(op, 'SQUARE')
    op = Operator(shapein=2)
    assert_flags(op, 'SQUARE')
    assert op.shapeout == op.shapein

    class Op(Operator):
        def reshapein(self, shape):
            if shape is None: return None
            return shape[0]*2
    op = Op()
    assert_flags_false(op, 'SQUARE')
    op = Op(shapein=2)
    assert_flags_false(op, 'SQUARE')
    assert op.shapeout == (4,)
    
    assert_raises(ValueError, Op, shapein=3, shapeout=11)

    op = Op(shapein=2) * Operator(shapein=4, shapeout=2)
    assert_flags(op, 'SQUARE')

def test_dtype1():
    value = 2.5
    @square
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
            yield assert_equal, o.dtype, (i*np.array(value,dop)).dtype, str((dop,di))
            yield assert_array_equal, o, i*np.array(value,dop), str((dop,di))


def test_dtype2():
    @square
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
        yield assert_equal, o.dtype, (i * i).dtype, str(di)
        yield assert_array_equal, o, i * i, str(di)


def test_symmetric():    
    mat = np.matrix([[2,1],[1,2]])
    @symmetric
    class Op(Operator):
        def __init__(self):
            Operator.__init__(self, shapein=(2,), dtype=mat.dtype)
        def direct(self, input, output):
            output[...] = np.dot(mat, input)

    op = Op()
    assert_flags(op, 'LINEAR,SQUARE,REAL,SYMMETRIC')
    assert_equal(op.shape, (2,2))
    assert_equal(op.shapeout, (2,))
    assert op is op.C
    assert op is op.T
    assert op is op.H
    assert_array_equal(op([1,1]), np.array(mat * [[1],[1]]).ravel())


def test_scalar_reduction1():
    models = 1.+I, -I, (-2) * I, -(2*I), 1.-I, 1.-2*I
    results = [6, -3, -6, -6, 0, -3]
    for model, result in zip(models, results):
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            o = model(i)
            yield assert_equal, o, result, str((model,i))
            yield assert_equal, o.dtype, np.float64, str((model,i))


def test_scalar_reduction2():
    model = -I
    iops = '__iadd__', '__isub__', '__imul__', '__iadd__', '__imul__'
    imodels = 2*I, 2*I, 2, O, O
    results = [3, -3, -6, -6, 0]
    for iop, imodel, result in zip(iops, imodels, results):
        model = getattr(model, iop)(imodel)
        for i in (np.array(3), [3], (3,), np.int(3), 3):
            yield eq_, model(i), result, str((iop,i))


def test_scalar_reduction3():
    for opout in (None, 100, 200):
        for opin in (None, 100, 200):
            for idin in (None, 100, 200):
                if opin is not None and idin is not None and opin != idin:
                    continue
                p = Operator(shapeout=opout, shapein=opin,
                    flags={'LINEAR':True}) * IdentityOperator(shapein=idin)

                n = len(p.operands) if isinstance(p, CompositionOperator) else 1
                if idin is None:
                    yield assert_equal, n, 1, str((opout,opin,idin,p))
                    continue
                if opin is None:
                    yield assert_equal, n, 2, str((opout,opin,idin,p))
                    yield ok_, isinstance(p.operands[1], IdentityOperator), \
                        str((opout,opin,idin,p))
                    continue
                if opin == idin and opout == idin:
                    yield assert_equal, n, 1, str((opout,opin,idin,p))


def test_propagation_attribute1():
    @square
    class AddAttribute(Operator):
        attrout = {'newattr_direct':True}
        attrin = {'newattr_transpose':True}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    @square
    class AddAttribute2(Operator):
        attrout = {'newattr_direct':False}
        attrin = {'newattr_transpose':False}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    @square
    class AddAttribute3(Operator):
        attrout = {'newattr3_direct':True}
        attrin = {'newattr3_transpose':True}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    inputs = [np.ones(5), np.ones(5).view(ndarray2)]
    for i in inputs:
        op = AddAttribute()
        yield ok_, op(i).newattr_direct
        yield ok_, op.T(i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        yield ok_, not op(i).newattr_direct
        yield ok_, op.T(i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        yield ok_, op(i).newattr_direct
        yield ok_, op(i).newattr3_direct
        yield ok_, op.T(i).newattr_transpose
        yield ok_, op.T(i).newattr3_transpose

    for i_ in inputs:
        op = AddAttribute()
        i = i_.copy()
        yield ok_, op(i,i).newattr_direct
        i = i_.copy()
        yield ok_, op.T(i,i).newattr_transpose

        op = AddAttribute2() * AddAttribute()
        i = i_.copy()
        yield ok_, not op(i,i).newattr_direct
        i = i_.copy()
        yield ok_, op.T(i,i).newattr_transpose

        op = AddAttribute3() * AddAttribute()
        i = i_.copy()
        o = op(i,i)
        yield ok_, o.newattr_direct
        yield ok_, o.newattr3_direct
        i = i_.copy()
        o = op.T(i,i)
        yield ok_, o.newattr_transpose
        yield ok_, o.newattr3_transpose


def test_propagation_attribute2():
    class Op(Operator):
        attrin = {'attr_class':1, 'attr_instance':2, 'attr_other':3}
        attrout = {'attr_class':4, 'attr_instance':5, 'attr_other':6}
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass
    class ndarray2(np.ndarray):
        attr_class = 10
        def __new__(cls, data):
            result = np.ndarray(data).view(cls)
            result.attr_instance = 11
            return result

    op = Op()
    output = op(ndarray2(1))
    assert output.__dict__ == op.attrout
    output = op.T(ndarray2(1))
    assert output.__dict__ == op.attrin


def test_propagation_attribute3():
    class ndarraybase(np.ndarray):
        attr_class = None
        def __new__(cls, data):
            result = np.array(data).view(cls)
            return result
        def __array_finalize__(self, array):
            self.attr_class = 0
            self.attr_instance = 10
    class ndarray1(ndarraybase):
        attr_class1 = None
        def __new__(cls, data):
            result = ndarraybase(data).view(cls)
            return result
        def __array_finalize__(self, array):
            ndarraybase.__array_finalize__(self, array)
            self.attr_class1 = 1
            self.attr_instance1 = 11
    class ndarray2(ndarraybase):
        attr_class2 = None
        def __new__(cls, data):
            result = ndarraybase(data).view(cls)
            return result
        def __array_finalize__(self, array):
            ndarraybase.__array_finalize__(self, array)
            self.attr_class2 = 2
            self.attr_instance2 = 12
    class Op(Operator):
        classin = ndarray1
        classout = ndarray2
        def direct(self, input, output):
            pass
        def transpose(self, input, output):
            pass

    op = Op()
    input = ndarray1(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {'attr_instance':10, 'attr_instance1':11,
        'attr_instance2':12, 'attr_class':30, 'attr_class2':2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':42,
                               'attr_class':30, 'attr_class2':32}

    op = Op().T
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':41,
                               'attr_class':30, 'attr_class1':31}
    input = ndarray2(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {'attr_instance':10, 'attr_instance2':12,
        'attr_instance1':11, 'attr_class':30, 'attr_class1':1}

    op = Op().T * Op() # -> ndarray2 -> ndarray1
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':41,
                               'attr_class':30, 'attr_class1':1}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance1':11,
        'attr_instance2':42, 'attr_class':30, 'attr_class1':1}

    op = Op() * Op().T # -> ndarray1 -> ndarray2
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':12,
        'attr_instance1':41, 'attr_class':30, 'attr_class2':2}
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {'attr_instance':40, 'attr_instance2':42,
                               'attr_class':30, 'attr_class2':2}    
    
def check_propagation_class(op, i, c):
    o = op(i)
    assert_is(type(o), c)

def check_propagation_class_inplace(op, i, c):
    i = i.copy()
    op(i,i)
    assert_is(type(i), c)

def test_propagation_class():
    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    results = [[np.ndarray, ndarray2],
               [ndarraywrap, ndarray2],
               [ndarray3, ndarray3],
               [ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op, i, c

def test_propagation_class_inplace():
    inputs = [np.ones(2), np.ones(2).view(ndarray2), np.ones(2).view(ndarray3)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    results = [[np.ndarray, ndarray2, ndarray3],
               [np.ndarray, ndarray2, ndarray3],
               [np.ndarray, ndarray3, ndarray3],
               [np.ndarray, ndarray3, ndarray3]]

    for op, results_ in zip(ops, results):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op, i, c

def test_propagation_classT():
    inputs = [np.ones(2), np.ones(2).view(ndarray2)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    resultsT = [[np.ndarray, ndarray2],
                [np.ndarray, ndarray2],
                [ndarray4, ndarray4],
                [ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class, op.T, i, c

def test_propagation_classT_inplace():
    inputs = [np.ones(2), np.ones(2).view(ndarray2), np.ones(2).view(ndarray4)]
    ops = [I, Op2(), Op2()*Op3(), Op3()*Op2()]
    resultsT = [[np.ndarray, ndarray2, ndarray4],
                [np.ndarray, ndarray2, ndarray4],
                [np.ndarray, ndarray4, ndarray4],
                [np.ndarray, ndarray4, ndarray4]]

    for op, results_ in zip(ops, resultsT):
        for i, c in zip(inputs, results_):
            yield check_propagation_class_inplace, op.T, i, c

def test_propagation_class_nested():
    class O1(Operator):
        classout = ndarray2
        def direct(self, input, output):
            output[...] = input
    class O2(Operator):
        def direct(self, input, output):
            output[...] = input

    def func2(op1, op2, expected):
        assert_is((op1*op2)(1).__class__, expected)

    def func3(op1, op2, op3, expected):
        assert_is((op1*op2*op3)(1).__class__, expected)

    o1 = O1()
    o2 = O2()
    ops1 = [I, 2*I, o2, 2*I+o2]
    ops2 = [I+o1, 2*o1, o1+o2, o2+o1, I+2*o1, I+o1+o2, I+o2+o1, o1+I+o2,
            o1+o2+I, o2+o1+I, o2+I+o1]
    for op1 in ops1:
        for op2 in ops2:
            yield func2, op1, op2, ndarray2
            yield func2, op2, op1, ndarray2
    for op1 in ops1:
        for op2 in ops2:
            for op3 in ops1:
                yield func3, op1, op2, op3, ndarray2

def test_inplace1():
    memory.stack = []
    @square
    class NotInplace(Operator):
        def direct(self, input, output):
            output[...] = 0
            output[0] = input[0]
    op = NotInplace()
    v = np.array([2., 0., 1.])
    op(v,v)
    assert_equal(v,[2,0,0])
    assert_equal(len(memory.stack), 1)


def test_inplace_can_use_output():

    A = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0] : 'A',
           B.__array_interface__['data'][0] : 'B',
           C.__array_interface__['data'][0] : 'C',
           D.__array_interface__['data'][0] : 'D',
           }
    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self)
            self.inplace = inplace
            self.log = log
        def direct(self, input, output):
            if self.inplace:
                tmp = input[0]
                output[1:] = 2 * input
                output[0] = tmp
            else:
                output[...] = 0
                output[0] = input[0]
                output[1:] = 2 * input
            try:
                self.log.insert(0, ids[output.__array_interface__['data'][0]])
            except KeyError:
                self.log.insert(0, 'unknown')
        def reshapein(self, shape):
            if shape is None:
                return None
            return (shape[0]+1)

    def show_stack():
        return ''.join([ids[s.__array_interface__['data'][0]] \
                            for s in memory.stack])

    expecteds_outplace = {
        2 : ['BBA',   #II
             'BBA',   #IO
             'BCA',   #OI
             'BCA'],  #OO
        3 : ['BBBA',  #III
             'BBBA',  #IIO
             'BBCA',  #IOI
             'BBCA',  #IOO
             'BCCA',  #OII
             'BCCA',  #OIO
             'BCBA',  #OOI
             'BCBA'], #OOO
        4 : ['BBBBA', #IIII
             'BBBBA', #IIIO
             'BBBCA', #IIOI
             'BBBCA', #IIOO
             'BBCCA', #IOII
             'BBCCA', #IOIO
             'BBCBA', #IOOI
             'BBCBA', #IOOO
             'BCCCA', #OIII
             'BCCCA', #OIIO
             'BCCBA', #OIOI
             'BCCBA', #OIOO
             'BCBBA', #OOII
             'BCBBA', #OOIO
             'BCBCA', #OOOI
             'BCBCA']}#OOOO

    expecteds_inplace = {
        2 : ['AAA',   #II
             'ABA',   #IO
             'ABA',   #OI
             'ABA'],  #OO
        3 : ['AAAA',  #III
             'ABBA',  #IIO
             'ABAA',  #IOI
             'AABA',  #IOO
             'ABAA',  #OII
             'ABBA',  #OIO
             'ABAA',  #OOI
             'ACBA'], #OOO
        4 : ['AAAAA', #IIII
             'ABBBA', #IIIO
             'ABBAA', #IIOI
             'AAABA', #IIOO
             'ABAAA', #IOII
             'AABBA', #IOIO
             'AABAA', #IOOI
             'ABABA', #IOOO
             'ABAAA', #OIII
             'ABBBA', #OIIO
             'ABBAA', #OIOI
             'ABABA', #OIOO
             'ABAAA', #OOII
             'ABABA', #OOIO
             'ABABA', #OOOI
             'ABABA']}#OOOO

    def func_outplace(n, i, expected, strops):
        memory.stack = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:8].view(float); v[0] = 1
        w = B[:(n+1)*8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected, strops)
        assert_equal(show_stack(), 'CD', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_array_equal(w, w2, strops)

    def func_inplace(n, i, expected, strops):
        memory.stack = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:8].view(float); v[0] = 1
        w = A[:(n+1)*8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected, strops)
        assert_equal(show_stack(), 'BC', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_array_equal(w, w2, strops)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, n, i, expected, strops

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, n, i, expected, strops

def test_inplace_cannot_use_output():

    A = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    B = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    C = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    D = np.zeros(10*8, dtype=np.int8).view(ndarraywrap)
    ids = {A.__array_interface__['data'][0] : 'A',
           B.__array_interface__['data'][0] : 'B',
           C.__array_interface__['data'][0] : 'C',
           D.__array_interface__['data'][0] : 'D',
           }
    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self)
            self.inplace = inplace
            self.log = log
        def direct(self, input, output):
            if not self.inplace:
                output[...] = 0
            output[:] = input[1:]
            try:
                self.log.insert(0, ids[output.__array_interface__['data'][0]])
            except KeyError:
                self.log.insert(0, 'unknown')
        def reshapein(self, shape):
            if shape is None:
                return None
            return (shape[0]-1)

    def show_stack():
        return ''.join([ids[s.__array_interface__['data'][0]] \
                            for s in memory.stack])

    expecteds_outplace = {
        2 : ['BCA',   #II
             'BCA',   #IO
             'BCA',   #OI
             'BCA'],  #OO
        3 : ['BCCA',  #III
             'BCCA',  #IIO
             'BDCA',  #IOI
             'BDCA',  #IOO
             'BCCA',  #OII
             'BCCA',  #OIO
             'BDCA',  #OOI
             'BDCA'], #OOO
        4 : ['BCCCA', #IIII
             'BCCCA', #IIIO
             'BDDCA', #IIOI
             'BDDCA', #IIOO
             'BDCCA', #IOII
             'BDCCA', #IOIO
             'BCDCA', #IOOI
             'BCDCA', #IOOO
             'BCCCA', #OIII
             'BCCCA', #OIIO
             'BDDCA', #OIOI
             'BDDCA', #OIOO
             'BDCCA', #OOII
             'BDCCA', #OOIO
             'BCDCA', #OOOI
             'BCDCA']}#OOOO

    expecteds_inplace = {
        2 : ['ABA',   #II
             'ABA',   #IO
             'ABA',   #OI
             'ABA'],  #OO
        3 : ['ABBA',  #III
             'ABBA',  #IIO
             'ACBA',  #IOI
             'ACBA',  #IOO
             'ABBA',  #OII
             'ABBA',  #OIO
             'ACBA',  #OOI
             'ACBA'], #OOO
        4 : ['ABBBA', #IIII
             'ABBBA', #IIIO
             'ACCBA', #IIOI
             'ACCBA', #IIOO
             'ACBBA', #IOII
             'ACBBA', #IOIO
             'ABCBA', #IOOI
             'ABCBA', #IOOO
             'ABBBA', #OIII
             'ABBBA', #OIIO
             'ACCBA', #OIOI
             'ACCBA', #OIOO
             'ACBBA', #OOII
             'ACBBA', #OOIO
             'ABCBA', #OOOI
             'ABCBA']}#OOOO

    def func_outplace(n, i, expected, strops):
        memory.stack = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float); v[:] = range(n+1)
        w = B[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected, strops)
        assert_equal(show_stack(), 'CD', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_array_equal(w, w2, strops)

    def func_inplace(n, i, expected, strops):
        memory.stack = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[:(n+1)*8].view(float); v[:] = range(n+1)
        w = A[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected, strops)
        assert_equal(show_stack(), 'BC', strops)
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_array_equal(w, w2, strops)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, n, i, expected, strops

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, n, i, expected, strops

    
def test_addition():
    @square
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    memory.stack = []
    op = np.sum([Op(v) for v in [1]])
    yield assert_is, op.__class__, Op

    op = np.sum([Op(v) for v in [1,2]])
    assert_equal(op.__class__, AdditionOperator)
    assert_array_equal(op(1), 3)
    assert_equal(len(memory.stack), 1)

    op = np.sum([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, AdditionOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    assert_array_equal(op(input, output), 7)
    assert_array_equal(input, 1)
    assert_array_equal(output, 7)
    assert_equal(len(memory.stack), 1)

    output = input
    assert_array_equal(op(input, output), 7)
    assert_array_equal(input, 7)
    assert_array_equal(output, 7)
    assert_equal(len(memory.stack), 2)


def test_composition1():
    s1 = ScalarOperator(2)
    s2 = ScalarOperator(3)
    assert s1(s2).data == 6

def test_composition2():
    @square
    @inplace
    class Op(Operator):
        def __init__(self, v, **keywords):
            self.v = v
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            np.multiply(input, self.v, output)

    memory.stack = []
    op = np.product([Op(v) for v in [1]])
    assert_is(op.__class__, Op)
    op(1)
    assert_equal(len(memory.stack), 0)

    memory.stack = []
    op = np.product([Op(v) for v in [1,2]])
    assert_is(op.__class__, CompositionOperator)
    assert_array_equal(op(1), 2)
    assert_equal(len(memory.stack), 0)

    memory.stack = []
    assert_array_equal(op([1]), 2)
    assert_equal(len(memory.stack), 0)

    op = np.product([Op(v) for v in [1,2,4]])
    assert_is(op.__class__, CompositionOperator)

    input = np.array(1, int)
    output = np.array(0, int)
    memory.stack = []
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 1)
    assert_array_equal(output, 8)
    assert_equal(len(memory.stack), 0)

    output = input
    memory.stack = []
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 8)
    assert_array_equal(output, 8)
    assert_equal(len(memory.stack), 0)

    input = np.array([1], int)
    output = np.array([0], int)
    memory.stack = []
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 1)
    assert_array_equal(output, 8)
    assert_equal(len(memory.stack), 0)

    output = input
    memory.stack = []
    assert_array_equal(op(input, output), 8)
    assert_array_equal(input, 8)
    assert_array_equal(output, 8)
    assert_equal(len(memory.stack), 0)


def test_scalar_operator():
    s = ScalarOperator(1)
    yield ok_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(-1)
    yield ok_, s.C is s.T is s.H is s.I is s

    s = ScalarOperator(2.)
    yield ok_, s.C is s.T is s.H is s
    yield assert_is_not, s.I, s
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_is_instance, o, ScalarOperator

    s = ScalarOperator(complex(0,1))
    yield assert_is, s.T, s
    yield assert_is, s.H, s.C
    yield assert_not_in, s.I, (s, s.C)
    yield assert_not_in, s.I.C, (s, s.C)
    yield assert_is_instance, s.C, ScalarOperator
    for o in (s.I, s.I.C, s.I.T, s.I.H, s.I.I):
        yield assert_is_instance, o, ScalarOperator

def test_partition1():
    o1 = ScalarOperator(1, shapein=1)
    o2 = ScalarOperator(2, shapein=2)
    o3 = ScalarOperator(3, shapein=3)
    
    r = DiagonalOperator([1,2,2,3,3,3]).todense()
    for ops, p in zip(((o1,o2,o3), (I,o2,o3), (o1,2*I,o3), (o1,o2,3*I)),
                      (None, (1,2,3), (1,2,3), (1,2,3))):
        op = PartitionOperator(ops, partitionin=p)
        yield assert_array_equal, op.todense(6), r, str(op)

def test_partition2():
    class Op(Operator):
        def __init__(self, axis, **keywords):
            self.axis = axis
            if self.axis < 0:
                self.slice = [Ellipsis] + (-self.axis) * [slice(None)]
            else:
                self.slice = (self.axis+1) * [slice(None)] + [Ellipsis]
            Operator.__init__(self, **keywords)
        def direct(self, input, output):
            self.slice[self.axis] = slice(0,None,2)
            output[self.slice] = input
            self.slice[self.axis] = slice(1,None,2)
            output[self.slice] = input
        def reshapein(self, shape):
            if shape is None: return None
            shape_ = list(shape)
            shape_[self.axis] *= 2
            return shape_
        def reshapeout(self, shape):
            if shape is None: return None
            shape_ = list(shape)
            shape_[self.axis] //= 2
            return shape_
    i = np.arange(3*4*5*6).reshape(3,4,5,6)
    for axisp,p in zip((0,1,2,3,-1,-2,-3), ((1,1,1),(1,2,1),(2,2,1),(2,3,1),(2,3,1),(2,2,1),(1,2,1),(1,1,1))):
        for axisr in (0,1,2,3):
            op = PartitionOperator(3*[Op(axisr)], partitionin=p, axisin=axisp)
            yield assert_array_equal, op(i), Op(axisr)(i), 'axis={},{}'.format(
                axisp,axisr)

def test_partition3():
    # test axisin != axisout...
    pass


def test_partition4():
    o1 = ScalarOperator(1, shapein=1)
    o2 = ScalarOperator(2, shapein=2)
    o3 = ScalarOperator(3, shapein=3)
    class Op(Operator):
        pass
    op=Op()
    p=PartitionOperator([o1,o2,o3])
    
    r = (op + p + op) * p
    assert isinstance(r, PartitionOperator)

def test_expansion1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, ExpansionOperator, [I2, 2*I3])

def test_expansion2():
    p = np.matrix([[1,0], [0,2], [1,0]])
    o = asoperator(np.matrix(p))
    e = ExpansionOperator([o, 2*o])
    assert_array_equal(e.todense(), np.vstack([p,2*p]))
    assert_array_equal(e.T.todense(), e.todense().T)

def test_reduction1():
    I2 = IdentityOperator(2)
    I3 = IdentityOperator(3)
    assert_raises(ValueError, ReductionOperator, [I2, 2*I3])

def test_reduction2():
    p = np.matrix([[1,0], [0,2], [1,0]])
    o = asoperator(np.matrix(p))
    r = ReductionOperator([o, 2*o])
    assert_array_equal(r.todense(), np.hstack([p,2*p]))
    assert_array_equal(r.T.todense(), r.todense().T)


if __name__ == "__main__":
    nose.run(argv=['', __file__])
