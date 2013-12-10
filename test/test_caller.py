from __future__ import division

import numpy as np
from nose import with_setup
from numpy.testing import assert_equal
from pyoperators import Operator, CompositionOperator, memory
from pyoperators.core import _pool as pool
from pyoperators.flags import square
from pyoperators.memory import zeros
from pyoperators.utils import isalias

old_memory_verbose = None
old_memory_tolerance = None


def test_inplace():
    @square
    class NotInplace(Operator):
        def direct(self, input, output):
            output[...] = 0
            output[0] = input[0]

    pool.clear()
    op = NotInplace()
    v = np.array([2.0, 0.0, 1.0])
    assert_equal(op(v), [2, 0, 0])
    op(v, v)
    assert_equal(v, [2, 0, 0])
    assert_equal(len(pool), 1)


def setup_memory():
    global old_memory_tolerance, old_memory_verbose
    old_memory_tolerance = memory.MEMORY_TOLERANCE
    old_memory_verbose = memory.verbose
    # ensure buffers in the pool are always used
    memory.MEMORY_TOLERANCE = np.inf
    memory.verbose = True


def teardown_memory():
    memory.MEMORY_TOLERANCE = old_memory_tolerance
    memory.verbose = old_memory_verbose


@with_setup(setup_memory, teardown_memory)
def test_inplace_can_use_output():
    A = zeros(10 * 8, dtype=np.int8)
    B = zeros(10 * 8, dtype=np.int8)
    C = zeros(10 * 8, dtype=np.int8)
    D = zeros(10 * 8, dtype=np.int8)
    ids = {
        A.__array_interface__['data'][0]: 'A',
        B.__array_interface__['data'][0]: 'B',
        C.__array_interface__['data'][0]: 'C',
        D.__array_interface__['data'][0]: 'D',
    }

    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace': inplace})
            self.inplace = inplace
            self.log = log

        def direct(self, input, output):
            if isalias(input, output):
                if not self.inplace:
                    raise RuntimeError()
                tmp = input[0]
                output[1:] = 2 * input
                output[0] = tmp
            else:
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

    expecteds_outplace = {
        2: ['BBA', 'BBA', 'BCA', 'BCA'],  # II  # IO  # OI  # OO
        3: [
            'BBBA',  # III
            'BBBA',  # IIO
            'BBCA',  # IOI
            'BBCA',  # IOO
            'BCCA',  # OII
            'BCCA',  # OIO
            'BCBA',  # OOI
            'BCBA',
        ],  # OOO
        4: [
            'BBBBA',  # IIII
            'BBBBA',  # IIIO
            'BBBCA',  # IIOI
            'BBBCA',  # IIOO
            'BBCCA',  # IOII
            'BBCCA',  # IOIO
            'BBCBA',  # IOOI
            'BBCBA',  # IOOO
            'BCCCA',  # OIII
            'BCCCA',  # OIIO
            'BCCBA',  # OIOI
            'BCCBA',  # OIOO
            'BCBBA',  # OOII
            'BCBBA',  # OOIO
            'BCBCA',  # OOOI
            'BCBCA',
        ],
    }  # OOOO

    expecteds_inplace = {
        2: ['AAA', 'ABA', 'ABA', 'ABA'],  # II  # IO  # OI  # OO
        3: [
            'AAAA',  # III
            'ABBA',  # IIO
            'ABAA',  # IOI
            'AABA',  # IOO
            'ABAA',  # OII
            'ABBA',  # OIO
            'ABAA',  # OOI
            'ACBA',
        ],  # OOO
        4: [
            'AAAAA',  # IIII
            'ABBBA',  # IIIO
            'ABBAA',  # IIOI
            'AAABA',  # IIOO
            'ABAAA',  # IOII
            'AABBA',  # IOIO
            'AABAA',  # IOOI
            'ABABA',  # IOOO
            'ABAAA',  # OIII
            'ABBBA',  # OIIO
            'ABBAA',  # OIOI
            'ABABA',  # OIOO
            'ABAAA',  # OOII
            'ABABA',  # OOIO
            'ABABA',  # OOOI
            'ABABA',
        ],
    }  # OOOO

    def func_outplace(strops, expected):
        n = len(strops)
        pool._buffers = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        v = A[:8].view(float)
        v[0] = 1
        w = B[: (n + 1) * 8].view(float)
        op(v, w)
        log = ''.join(log) + 'A'
        assert_equal(log, expected)
        assert_equal(show_pool(), 'CD')
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_equal(w, w2)

    def func_inplace(strops, expected):
        n = len(strops)
        pool._buffers = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        v = A[:8].view(float)
        v[0] = 1
        w = A[: (n + 1) * 8].view(float)
        op(v, w)
        log = ''.join(log) + 'A'
        assert_equal(log, expected)
        assert_equal(show_pool(), 'BC')
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_equal(w, w2)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, strops, expected

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, strops, expected


@with_setup(setup_memory, teardown_memory)
def test_inplace_cannot_use_output():
    A = zeros(10 * 8, dtype=np.int8)
    B = zeros(10 * 8, dtype=np.int8)
    C = zeros(10 * 8, dtype=np.int8)
    D = zeros(10 * 8, dtype=np.int8)
    ids = {
        A.__array_interface__['data'][0]: 'A',
        B.__array_interface__['data'][0]: 'B',
        C.__array_interface__['data'][0]: 'C',
        D.__array_interface__['data'][0]: 'D',
    }

    class Op(Operator):
        def __init__(self, inplace, log):
            Operator.__init__(self, flags={'inplace': inplace})
            self.inplace = inplace
            self.log = log

        def direct(self, input, output):
            if isalias(input, output) and not self.inplace:
                raise RuntimeError()
            output[:] = input[1:]
            try:
                self.log.insert(0, ids[output.__array_interface__['data'][0]])
            except KeyError:
                self.log.insert(0, '?')

        def reshapein(self, shape):
            return (shape[0] - 1,)

    def show_stack():
        return ''.join([ids[s.__array_interface__['data'][0]] for s in pool])

    expecteds_outplace = {
        2: ['BCA', 'BCA', 'BCA', 'BCA'],  # II  # IO  # OI  # OO
        3: [
            'BCCA',  # III
            'BCCA',  # IIO
            'BDCA',  # IOI
            'BDCA',  # IOO
            'BCCA',  # OII
            'BCCA',  # OIO
            'BDCA',  # OOI
            'BDCA',
        ],  # OOO
        4: [
            'BCCCA',  # IIII
            'BCCCA',  # IIIO
            'BDDCA',  # IIOI
            'BDDCA',  # IIOO
            'BDCCA',  # IOII
            'BDCCA',  # IOIO
            'BCDCA',  # IOOI
            'BCDCA',  # IOOO
            'BCCCA',  # OIII
            'BCCCA',  # OIIO
            'BDDCA',  # OIOI
            'BDDCA',  # OIOO
            'BDCCA',  # OOII
            'BDCCA',  # OOIO
            'BCDCA',  # OOOI
            'BCDCA',
        ],
    }  # OOOO

    expecteds_inplace = {
        2: ['ABA', 'ABA', 'ABA', 'ABA'],  # II  # IO  # OI  # OO
        3: [
            'ABBA',  # III
            'ABBA',  # IIO
            'ACBA',  # IOI
            'ACBA',  # IOO
            'ABBA',  # OII
            'ABBA',  # OIO
            'ACBA',  # OOI
            'ACBA',
        ],  # OOO
        4: [
            'ABBBA',  # IIII
            'ABBBA',  # IIIO
            'ACCBA',  # IIOI
            'ACCBA',  # IIOO
            'ACBBA',  # IOII
            'ACBBA',  # IOIO
            'ABCBA',  # IOOI
            'ABCBA',  # IOOO
            'ABBBA',  # OIII
            'ABBBA',  # OIIO
            'ACCBA',  # OIOI
            'ACCBA',  # OIOO
            'ACBBA',  # OOII
            'ACBBA',  # OOIO
            'ABCBA',  # OOOI
            'ABCBA',
        ],
    }  # OOOO

    def func_outplace(strops, expected):
        n = len(strops)
        pool._buffers = [C, D]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[: (n + 1) * 8].view(float)
        v[:] = range(n + 1)
        w = B[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected)
        assert_equal(show_stack(), 'CD')
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_equal(w, w2)

    def func_inplace(strops, expected):
        n = len(strops)
        pool._buffers = [B, C]
        log = []
        ops = [Op(s == '1', log) for s in strops]
        op = CompositionOperator(ops)
        op.show_stack = show_stack
        v = A[: (n + 1) * 8].view(float)
        v[:] = range(n + 1)
        w = A[:8].view(float)
        op(v, w)
        delattr(op, 'show_stack')
        log = ''.join(log) + 'A'
        assert_equal(log, expected)
        assert_equal(show_stack(), 'BC')
        w2 = v
        for op in reversed(ops):
            w2 = op(w2)
        assert_equal(w, w2)

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_outplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_outplace, strops, expected

    for n in [2, 3, 4]:
        for i, expected in zip(reversed(range(2**n)), expecteds_inplace[n]):
            strops = bin(i)[2:]
            while len(strops) != n:
                strops = '0' + strops
            yield func_inplace, strops, expected
