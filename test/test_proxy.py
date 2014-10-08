from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from pyoperators import AdditionOperator, CompositionOperator, Operator, flags
from pyoperators.utils.testing import assert_same
from pyoperators.proxy import proxy_group


mat = np.array([[1, 1, 1j],
                [0, 1, 1],
                [0, 0, 1]])
matI = np.linalg.inv(mat)

global counter
counter = 0


@flags.linear
@flags.square
class MyOperator(Operator):
    def __init__(self, i, **keywords):
        self.i = i
        Operator.__init__(self, dtype=np.array(mat).dtype, **keywords)

    def direct(self, x, out):
        out[...] = self.i * np.dot(mat, x)

    def conjugate(self, x, out):
        out[...] = self.i * np.dot(mat.conjugate(), x)

    def transpose(self, x, out):
        out[...] = self.i * np.dot(x, mat)

    def adjoint(self, x, out):
        out[...] = self.i * np.dot(x, mat.conjugate())

    def inverse(self, x, out):
        out[...] = 1 / self.i * np.dot(matI, x)

    def inverse_conjugate(self, x, out):
        out[...] = 1 / self.i * np.dot(matI.conjugate(), x)

    def inverse_transpose(self, x, out):
        out[...] = 1 / self.i * np.dot(x, matI)

    def inverse_adjoint(self, x, out):
        out[...] = 1 / self.i * np.dot(x, matI.conjugate())


def callback(i):
    global counter
    counter += 1
    return MyOperator(i + 1, shapein=3)


def get_operator(list, attr):
    if attr == '':
        return list
    elif attr == 'IC':
        return [_.I.C for _ in list]
    elif attr == 'IT':
        return [_.I.T for _ in list]
    elif attr == 'IH':
        return [_.I.H for _ in list]
    return [getattr(_, attr) for _ in list]


nproxy = 5
ref_list = [callback(i) for i in range(nproxy)]
proxy_list = proxy_group(nproxy, callback)


def test_copy():
    proxy = proxy_list[0]
    assert proxy.copy().common is proxy.common


def test():
    def func(attr):
        olist = get_operator(proxy_list, attr)
        rlist = get_operator(ref_list, attr)
        for o, r in zip(olist, rlist):
            assert_same(o.todense(), r.todense())
    for attr in '', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH':
        yield func, attr


def test_addition():
    def func(attr):
        op = AdditionOperator(get_operator(proxy_list, attr))
        ref = AdditionOperator(get_operator(ref_list, attr))
        assert_same(op.todense(), ref.todense())
    for attr in '', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH':
        yield func, attr


def test_composite():
    global counter
    counter = 0
    proxy_lists = [get_operator(proxy_list, attr)
                   for attr in ('', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH')]
    ref_lists = [get_operator(ref_list, attr)
                 for attr in ('', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH')]

    op = AdditionOperator(CompositionOperator(_) for _ in zip(*proxy_lists))
    ref = AdditionOperator(CompositionOperator(_) for _ in zip(*ref_lists))
    assert_same(op.todense(), ref.todense())
    assert_equal(counter, nproxy * op.shapein[0])


def test_getattr():
    assert_equal(np.sum(_.i for _ in proxy_list),
                 np.sum(_.i for _ in ref_list))
