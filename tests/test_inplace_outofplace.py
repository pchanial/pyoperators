import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import CompositionOperator, Operator, flags
from pyoperators.core import _pool as pool
from pyoperators.memory import zeros
from pyoperators.utils import isalias, ndarraywrap
from pyoperators.utils.testing import assert_eq


def test_inplace():
    @flags.square
    class NotInplace(Operator):
        def direct(self, input, output):
            output[...] = 0
            output[0] = input[0]

    pool.clear()
    op = NotInplace()
    v = np.array([2.0, 0.0, 1.0])
    op(v, v)
    assert_equal(v, [2, 0, 0])
    assert_equal(len(pool), 1)


@pytest.fixture
def always_reuse_buffer(mocker):
    # ensure buffers in the pool are always used
    mocker.patch('pyoperators.config.MEMORY_TOLERANCE', np.inf)
    mocker.patch('pyoperators.config.VERBOSE', True)
    yield


A = zeros(10 * 8, dtype=np.int8).view(ndarraywrap)
B = zeros(10 * 8, dtype=np.int8).view(ndarraywrap)
C = zeros(10 * 8, dtype=np.int8).view(ndarraywrap)
D = zeros(10 * 8, dtype=np.int8).view(ndarraywrap)
ids = {
    A.__array_interface__['data'][0]: 'A',
    B.__array_interface__['data'][0]: 'B',
    C.__array_interface__['data'][0]: 'C',
    D.__array_interface__['data'][0]: 'D',
}


class OpCanUseOutput(Operator):
    def __init__(self, inplace, log):
        Operator.__init__(self, flags={'inplace': inplace})
        self.inplace = inplace
        self.log = log

    def direct(self, input, output):
        if not self.inplace and isalias(input, output):
            pytest.xfail('Unknown.')
        if self.flags.inplace:
            tmp = input[0]
            output[1:] = 2 * input
            output[0] = tmp
        else:
            output[:] = 0
            output[0] = input[0]
            output[1:] = 2 * input
        try:
            self.log.insert(0, ids[output.__array_interface__['data'][0]])
        except KeyError:
            self.log.insert(0, '?')

    def reshapein(self, shape):
        return (shape[0] + 1,)


def show_pool():
    result = ''
    for s in pool:
        try:
            result += ids[s.__array_interface__['data'][0]]
        except:
            result += '?'
    return result


EXPECTED_OUTPLACE_CAN_USE_OUTPUT = [
    ('II', 'BBA'),
    ('IO', 'BBA'),
    ('OI', 'BCA'),
    ('OO', 'BCA'),
    ('III', 'BBBA'),
    ('IIO', 'BBBA'),
    ('IOI', 'BBCA'),
    ('IOO', 'BBCA'),
    ('OII', 'BCCA'),
    ('OIO', 'BCCA'),
    ('OOI', 'BCBA'),
    ('OOO', 'BCBA'),
    ('IIII', 'BBBBA'),
    ('IIIO', 'BBBBA'),
    ('IIOI', 'BBBCA'),
    ('IIOO', 'BBBCA'),
    ('IOII', 'BBCCA'),
    ('IOIO', 'BBCCA'),
    ('IOOI', 'BBCBA'),
    ('IOOO', 'BBCBA'),
    ('OIII', 'BCCCA'),
    ('OIIO', 'BCCCA'),
    ('OIOI', 'BCCBA'),
    ('OIOO', 'BCCBA'),
    ('OOII', 'BCBBA'),
    ('OOIO', 'BCBBA'),
    ('OOOI', 'BCBCA'),
    ('OOOO', 'BCBCA'),
]


@pytest.mark.parametrize('op_kinds, expected', EXPECTED_OUTPLACE_CAN_USE_OUTPUT)
def test_inplace_can_use_output_check_outplace(always_reuse_buffer, op_kinds, expected):
    nops = len(op_kinds)
    pool._buffers = [C, D]
    log = []
    ops = [OpCanUseOutput(s == 'I', log) for s in op_kinds]
    op = CompositionOperator(ops)
    op.show_pool = show_pool  # debug
    v = A[:8].view(float)
    v[0] = 1
    w = B[: (nops + 1) * 8].view(float)
    op(v, w)
    log = ''.join(log) + 'A'
    assert log == expected
    assert show_pool() == 'CD'
    w2 = v
    for op in reversed(ops):
        w2 = op(w2)
    assert_eq(w, w2)


EXPECTED_INPLACE_CAN_USE_OUTPUT = [
    ('II', 'AAA'),
    ('IO', 'ABA'),
    ('OI', 'ABA'),
    ('OO', 'ABA'),
    ('III', 'AAAA'),
    ('IIO', 'ABBA'),
    ('IOI', 'ABAA'),
    ('IOO', 'AABA'),
    ('OII', 'ABAA'),
    ('OIO', 'ABBA'),
    ('OOI', 'ABAA'),
    pytest.param('OOO', 'ACBA', marks=pytest.mark.xfail),
    ('IIII', 'AAAAA'),
    ('IIIO', 'ABBBA'),
    ('IIOI', 'ABBAA'),
    ('IIOO', 'AAABA'),
    ('IOII', 'ABAAA'),
    ('IOIO', 'AABBA'),
    ('IOOI', 'AABAA'),
    pytest.param('IOOO', 'ABABA', marks=pytest.mark.xfail),
    ('OIII', 'ABAAA'),
    ('OIIO', 'ABBBA'),
    ('OIOI', 'ABBAA'),
    pytest.param('OIOO', 'ABABA', marks=pytest.mark.xfail),
    ('OOII', 'ABAAA'),
    pytest.param('OOIO', 'ABABA', marks=pytest.mark.xfail),
    pytest.param('OOOI', 'ABABA', marks=pytest.mark.xfail),
    ('OOOO', 'ABABA'),
]


@pytest.mark.parametrize('op_kinds, expected', EXPECTED_INPLACE_CAN_USE_OUTPUT)
def test_inplace_can_use_output_check_inplace(always_reuse_buffer, op_kinds, expected):
    n = len(op_kinds)
    pool._buffers = [B, C]
    log = []
    ops = [OpCanUseOutput(s == 'I', log) for s in op_kinds]
    op = CompositionOperator(ops)
    v = A[:8].view(float)
    v[0] = 1
    w = A[: (n + 1) * 8].view(float)
    op(v, w)
    log = ''.join(log) + 'A'
    assert log == expected
    assert show_pool() == 'BC'
    w2 = v
    for op in reversed(ops):
        w2 = op(w2)
    assert_eq(w, w2)


class OpCannotUseOutput(Operator):
    def __init__(self, inplace, log):
        Operator.__init__(self, flags={'inplace': inplace})
        self.inplace = inplace
        self.log = log

    def direct(self, input, output):
        if not self.inplace and isalias(input, output):
            pytest.xfail('Unknown.')
        if not self.inplace:
            output[:] = 0
        output[:] = input[1:]
        try:
            self.log.insert(0, ids[output.__array_interface__['data'][0]])
        except KeyError:
            self.log.insert(0, '?')

    def reshapein(self, shape):
        return (shape[0] - 1,)


def show_stack():
    return ''.join([ids[s.__array_interface__['data'][0]] for s in pool])


EXPECTED_OUTPLACE_CANNOT_USE_OUTPUT = [
    ('II', 'BCA'),
    ('IO', 'BCA'),
    ('OI', 'BCA'),
    ('OO', 'BCA'),
    ('III', 'BCCA'),
    ('IIO', 'BCCA'),
    pytest.param('IOI', 'BDCA', marks=pytest.mark.xfail),
    pytest.param('IOO', 'BDCA', marks=pytest.mark.xfail),
    ('OII', 'BCCA'),
    ('OIO', 'BCCA'),
    pytest.param('OOI', 'BDCA', marks=pytest.mark.xfail),
    pytest.param('OOO', 'BDCA', marks=pytest.mark.xfail),
    ('IIII', 'BCCCA'),
    ('IIIO', 'BCCCA'),
    pytest.param('IIOI', 'BDDCA', marks=pytest.mark.xfail),
    pytest.param('IIOO', 'BDDCA', marks=pytest.mark.xfail),
    pytest.param('IOII', 'BDCCA', marks=pytest.mark.xfail),
    pytest.param('IOIO', 'BDCCA', marks=pytest.mark.xfail),
    ('IOOI', 'BCDCA'),
    ('IOOO', 'BCDCA'),
    ('OIII', 'BCCCA'),
    ('OIIO', 'BCCCA'),
    pytest.param('OIOI', 'BDDCA', marks=pytest.mark.xfail),
    pytest.param('OIOO', 'BDDCA', marks=pytest.mark.xfail),
    pytest.param('OOII', 'BDCCA', marks=pytest.mark.xfail),
    pytest.param('OOIO', 'BDCCA', marks=pytest.mark.xfail),
    ('OOOI', 'BCDCA'),
    ('OOOO', 'BCDCA'),
]


@pytest.mark.parametrize('op_kinds, expected', EXPECTED_OUTPLACE_CANNOT_USE_OUTPUT)
def test_inplace_cannot_use_output_check_outplace(
    always_reuse_buffer, op_kinds, expected
):
    nops = len(op_kinds)
    pool._buffers = [C, D]
    log = []
    ops = [OpCannotUseOutput(s == 'I', log) for s in op_kinds]
    op = CompositionOperator(ops)
    op.show_stack = show_stack
    v = A[: (nops + 1) * 8].view(float)
    v[:] = range(nops + 1)
    w = B[:8].view(float)
    op(v, w)
    delattr(op, 'show_stack')
    log = ''.join(log) + 'A'
    assert log == expected
    assert show_stack() == 'CD'
    w2 = v
    for op in reversed(ops):
        w2 = op(w2)
    assert_eq(w, w2)


EXPECTED_INPLACE_CANNOT_USE_OUTPUT = [
    ('II', 'ABA'),
    ('IO', 'ABA'),
    ('OI', 'ABA'),
    ('OO', 'ABA'),
    pytest.param('III', 'ABBA', marks=pytest.mark.xfail),
    pytest.param('IIO', 'ABBA', marks=pytest.mark.xfail),
    pytest.param('IOI', 'ACBA', marks=pytest.mark.xfail),
    pytest.param('IOO', 'ACBA', marks=pytest.mark.xfail),
    pytest.param('OII', 'ABBA', marks=pytest.mark.xfail),
    pytest.param('OIO', 'ABBA', marks=pytest.mark.xfail),
    pytest.param('OOI', 'ACBA', marks=pytest.mark.xfail),
    pytest.param('OOO', 'ACBA', marks=pytest.mark.xfail),
    pytest.param('IIII', 'ABBBA', marks=pytest.mark.xfail),
    pytest.param('IIIO', 'ABBBA', marks=pytest.mark.xfail),
    pytest.param('IIOI', 'ACCBA', marks=pytest.mark.xfail),
    pytest.param('IIOO', 'ACCBA', marks=pytest.mark.xfail),
    pytest.param('IOII', 'ACBBA', marks=pytest.mark.xfail),
    pytest.param('IOIO', 'ACBBA', marks=pytest.mark.xfail),
    ('IOOI', 'ABCBA'),
    ('IOOO', 'ABCBA'),
    pytest.param('OIIO', 'ABBBA', marks=pytest.mark.xfail),
    pytest.param('OIII', 'ABBBA', marks=pytest.mark.xfail),
    pytest.param('OIOI', 'ACCBA', marks=pytest.mark.xfail),
    pytest.param('OIOO', 'ACCBA', marks=pytest.mark.xfail),
    pytest.param('OOII', 'ACBBA', marks=pytest.mark.xfail),
    pytest.param('OOIO', 'ACBBA', marks=pytest.mark.xfail),
    ('OOOI', 'ABCBA'),
    ('OOOO', 'ABCBA'),
]


@pytest.mark.parametrize('op_kinds, expected', EXPECTED_INPLACE_CANNOT_USE_OUTPUT)
def test_inplace_cannot_use_output_check_inplace(
    always_reuse_buffer, op_kinds, expected
):
    nops = len(op_kinds)
    pool._buffers = [B, C]
    log = []
    ops = [OpCannotUseOutput(s == '1', log) for s in op_kinds]
    op = CompositionOperator(ops)
    op.show_stack = show_stack
    v = A[: (nops + 1) * 8].view(float)
    v[:] = range(nops + 1)
    w = A[:8].view(float)
    op(v, w)
    delattr(op, 'show_stack')
    log = ''.join(log) + 'A'
    assert log == expected
    assert show_stack() == 'BC'
    w2 = v
    for op in reversed(ops):
        w2 = op(w2)
    assert_eq(w, w2)
