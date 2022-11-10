import numpy as np
import pytest

from pyoperators import AdditionOperator, CompositionOperator, Operator, flags
from pyoperators.proxy import proxy_group
from pyoperators.utils.testing import assert_same

from .common import get_associated_operator

mat = np.array([[1, 1, 1j], [0, 1, 1], [0, 0, 1]])
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
        out[...] = self.i * (mat @ x)

    def conjugate(self, x, out):
        out[...] = self.i * (mat.conjugate() @ x)

    def transpose(self, x, out):
        out[...] = self.i * (x @ mat)

    def adjoint(self, x, out):
        out[...] = self.i * (x @ mat.conjugate())

    def inverse(self, x, out):
        out[...] = 1 / self.i * (matI @ x)

    def inverse_conjugate(self, x, out):
        out[...] = 1 / self.i * (matI.conjugate() @ x)

    def inverse_transpose(self, x, out):
        out[...] = 1 / self.i * (x @ matI)

    def inverse_adjoint(self, x, out):
        out[...] = 1 / self.i * (x @ matI.conjugate())


def callback(i):
    global counter
    counter += 1
    return MyOperator(i + 1, shapein=3)


nproxy = 5
ref_list = [callback(i) for i in range(nproxy)]
proxy_list = proxy_group(nproxy, callback)


def test_copy():
    proxy = proxy_list[0]
    assert proxy.copy().common is proxy.common


def get_associated_operators(list, attr):
    return [get_associated_operator(_, attr) for _ in list]


@pytest.mark.parametrize('attr', ['', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH'])
def test(attr):
    olist = get_associated_operators(proxy_list, attr)
    rlist = get_associated_operators(ref_list, attr)
    for o, r in zip(olist, rlist):
        assert_same(o.todense(), r.todense())


@pytest.mark.parametrize('attr', ['', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH'])
def test_addition(attr):
    op = AdditionOperator(get_associated_operators(proxy_list, attr))
    ref = AdditionOperator(get_associated_operators(ref_list, attr))
    assert_same(op.todense(), ref.todense())


def test_composite():
    global counter
    counter = 0
    proxy_lists = [
        get_associated_operators(proxy_list, attr)
        for attr in ('', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH')
    ]
    ref_lists = [
        get_associated_operators(ref_list, attr)
        for attr in ('', 'C', 'T', 'H', 'I', 'IC', 'IT', 'IH')
    ]

    op = AdditionOperator(CompositionOperator(_) for _ in zip(*proxy_lists))
    ref = AdditionOperator(CompositionOperator(_) for _ in zip(*ref_lists))
    assert_same(op.todense(), ref.todense())
    assert counter == nproxy * op.shapein[0]


def test_getattr():
    assert sum(_.i for _ in proxy_list) == sum(_.i for _ in ref_list)
