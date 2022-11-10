import numpy as np
import pytest

from pyoperators import Operator, flags


class ndarray2(np.ndarray):
    pass


@flags.linear
@flags.square
class OpAddAttribute(Operator):
    attrout = {'newattr_direct': True}
    attrin = {'newattr_transpose': True}

    def direct(self, input, output):
        pass

    def transpose(self, input, output):
        pass


@flags.linear
@flags.square
class OpAddAttribute2(Operator):
    attrout = {'newattr_direct': False}
    attrin = {'newattr_transpose': False}

    def direct(self, input, output):
        pass

    def transpose(self, input, output):
        pass


@flags.linear
@flags.square
class OpAddAttribute3(Operator):
    attrout = {'newattr3_direct': True}
    attrin = {'newattr3_transpose': True}

    def direct(self, input, output):
        pass

    def transpose(self, input, output):
        pass


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute1(input):
    op = OpAddAttribute()
    assert op(input).newattr_direct
    assert op.T(input).newattr_transpose


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute2(input):
    op = OpAddAttribute2() @ OpAddAttribute()
    assert not op(input).newattr_direct
    assert op.attrout == {'newattr_direct': False}
    assert op.attrin == {'newattr_transpose': True}
    assert op.T(input).newattr_transpose


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute3(input):
    op = OpAddAttribute3() @ OpAddAttribute()
    assert op(input).newattr_direct
    assert op(input).newattr3_direct
    assert op.attrout == {'newattr_direct': True, 'newattr3_direct': True}
    assert op.attrin == {'newattr_transpose': True, 'newattr3_transpose': True}
    assert op.T(input).newattr_transpose
    assert op.T(input).newattr3_transpose


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute4(input):
    op = OpAddAttribute()
    i = input.copy()
    assert op(i, i).newattr_direct
    i = input.copy()
    assert op.T(i, i).newattr_transpose


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute5(input):
    op = OpAddAttribute2() @ OpAddAttribute()
    i = input.copy()
    assert not op(i, i).newattr_direct
    i = input.copy()
    assert op.T(i, i).newattr_transpose


@pytest.mark.parametrize('input', [np.ones(5), np.ones(5).view(ndarray2)])
def test_add_attribute6(input):
    op = OpAddAttribute3() @ OpAddAttribute()
    i = input.copy()
    o = op(i, i)
    assert o.newattr_direct
    assert o.newattr3_direct
    i = input.copy()
    o = op.T(i, i)
    assert o.newattr_transpose
    assert o.newattr3_transpose


def test_propagation_attribute1():
    @flags.square
    class Op(Operator):
        attrin = {'attr_class': 1, 'attr_instance': 2, 'attr_other': 3}
        attrout = {'attr_class': 4, 'attr_instance': 5, 'attr_other': 6}

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


def test_propagation_attribute2():
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

    @flags.linear
    @flags.square
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
    assert output.__dict__ == {
        'attr_instance': 10,
        'attr_instance1': 11,
        'attr_instance2': 12,
        'attr_class': 30,
        'attr_class2': 2,
    }
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance2': 42,
        'attr_class': 30,
        'attr_class2': 32,
    }

    op = Op().T
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance1': 41,
        'attr_class': 30,
        'attr_class1': 31,
    }
    input = ndarray2(1)
    input.attr_class = 30
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 10,
        'attr_instance2': 12,
        'attr_instance1': 11,
        'attr_class': 30,
        'attr_class1': 1,
    }

    op = Op().T @ Op()  # -> ndarray2 -> ndarray1
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance1': 41,
        'attr_class': 30,
        'attr_class1': 1,
    }
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance1': 11,
        'attr_instance2': 42,
        'attr_class': 30,
        'attr_class1': 1,
    }

    op = Op() @ Op().T  # -> ndarray1 -> ndarray2
    input = ndarray1(1)
    input.attr_class = 30
    input.attr_class1 = 31
    input.attr_instance = 40
    input.attr_instance1 = 41
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance2': 12,
        'attr_instance1': 41,
        'attr_class': 30,
        'attr_class2': 2,
    }
    input = ndarray2(1)
    input.attr_class = 30
    input.attr_class2 = 32
    input.attr_instance = 40
    input.attr_instance2 = 42
    output = op(input)
    assert output.__dict__ == {
        'attr_instance': 40,
        'attr_instance2': 42,
        'attr_class': 30,
        'attr_class2': 2,
    }
