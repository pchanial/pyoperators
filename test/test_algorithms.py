from __future__ import division

import numpy as np

from pyoperators.iterative.core import IterativeAlgorithm
from pyoperators.iterative.stopconditions import MaxIterationStopCondition
from pyoperators.utils.testing import assert_eq, assert_raises, assert_same


def test_suffix():
    suffix = ['_new', '', '_old', '_old2', '_old3', '_old4']
    expecteds = [['']] + [suffix[:n] for n in range(2, len(suffix)+1)]
    keywords = [{'x': 0, 'inplace_recursion': True},
                {'x': 0},
                {'x': 0, 'x_old': 1},
                {'x': 0, 'x_old': 1, 'x_old2': 2},
                {'x': 0, 'x_old': 1, 'x_old2': 2, 'x_old3': 3},
                {'x': 0, 'x_old': 1, 'x_old2': 2, 'x_old3': 3, 'x_old4': 4}]

    def func(keywords, expected):
        a = IterativeAlgorithm(**keywords)
        assert_eq(a._get_suffix(), expected)
    for keyword, expected in zip(keywords, expecteds):
        yield func, keyword, expected


def test_fibonacci():
    class Fibonacci(IterativeAlgorithm):
        def __init__(self, **keywords):
            IterativeAlgorithm.__init__(self, x_old=0, x=1, **keywords)

        def iteration(self):
            np.add(self.x_old, self.x, self.x_new)

        def finalize(self):
            return int(self.x)

    fib = Fibonacci(normal_stop_condition=MaxIterationStopCondition(10))
    assert_eq(fib.run(), 55)
    fib.initialize()
    assert_eq(list(fib), [1, 2, 3, 5, 8, 13, 21, 34, 55])
    assert_eq(fib.restart(), 55)


def test_recursion():
    class GaussLegendre1(IterativeAlgorithm):
        def __init__(self, **keywords):
            IterativeAlgorithm.__init__(
                self, a=1, b=1/np.sqrt(2), t=1/4, p=1, p_dtype=int,
                normal_stop_condition=MaxIterationStopCondition(10),
                **keywords)

        def iteration(self):
            self.a_new[...] = (self.a + self.b) / 2
            self.b_new[...] = np.sqrt(self.a * self.b)
            self.t_new[...] = self.t - self.p*(self.a - self.a_new)**2
            self.p_new[...] = 2 * self.p

        def finalize(self):
            return (self.a + self.b)**2/(4*self.t)

    class GaussLegendre2(IterativeAlgorithm):
        def __init__(self, **keywords):
            IterativeAlgorithm.__init__(
                self, a=1, b=1/np.sqrt(2), t=1/4, p=1, p_dtype=int,
                allocate_new_state=False, normal_stop_condition=
                MaxIterationStopCondition(10), **keywords)

        def iteration(self):
            self.a_new = (self.a + self.b) / 2
            self.b_new = np.sqrt(self.a * self.b)
            self.t_new = self.t - self.p*(self.a - self.a_new)**2
            self.p_new = 2 * self.p

        def finalize(self):
            return (self.a + self.b)**2/(4*self.t)

    class GaussLegendre3(IterativeAlgorithm):
        def __init__(self, **keywords):
            IterativeAlgorithm.__init__(
                self, a=1, b=1/np.sqrt(2), t=1/4, p=1, p_dtype=int,
                inplace_recursion=True, normal_stop_condition=
                MaxIterationStopCondition(10), **keywords)

        def iteration(self):
            a_tmp = (self.a + self.b) / 2
            self.b[...] = np.sqrt(self.a * self.b)
            self.t -= self.p*(self.a - a_tmp)**2
            self.p *= 2
            self.a[...] = a_tmp

        def finalize(self):
            return (self.a + self.b)**2/(4*self.t)

    class GaussLegendre4(IterativeAlgorithm):
        def __init__(self, **keywords):
            IterativeAlgorithm.__init__(
                self, a=1, b=1/np.sqrt(2), t=1/4, p=1, p_dtype=int,
                inplace_recursion=True, normal_stop_condition=
                MaxIterationStopCondition(10), **keywords)

        def iteration(self):
            a_tmp = (self.a + self.b) / 2
            self.b = np.sqrt(self.a * self.b)
            self.t = self.t - self.p*(self.a - a_tmp)**2
            self.p = 2 * self.p
            self.a = a_tmp

        def finalize(self):
            return (self.a + self.b)**2/(4*self.t)

    algos = [GaussLegendre1, GaussLegendre2, GaussLegendre3, GaussLegendre4]

    def func(algo, reuse_initial_state):
        g = algo(reuse_initial_state=reuse_initial_state)
        pi = g.run()
        assert g.niterations == 10
        assert_same(pi, np.pi)
        if g.reuse_initial_state:
            assert_raises(RuntimeError, g.restart)
            return
        g.restart()
        assert g.niterations == 10
        assert_same(pi, np.pi)
    for algo in algos:
        for reuse_initial_state in (False, True):
            yield func, algo, reuse_initial_state
