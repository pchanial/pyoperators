from __future__ import absolute_import, division, print_function
import inspect
from . import core
from .utils import operation_assignment, strshape

__all__ = ['proxy_group']


class ProxyBase(core.Operator):
    def __init__(self, number, common, callback, flags, **keywords):
        if len(common) != 2:
            raise ValueError('Invalid common list for on-fly operators.')
        self.number = number
        self.common = common
        self.callback = callback
        core.Operator.__init__(self, flags=flags, **keywords)

    def get_operator(self):
        if self.common[0] != self.number:
            self.common[1].delete()
            self.common[:] = self.number, self.callback(self.number)
        return self.common[1]

    def direct(self, x, out, operation=operation_assignment):
        op = self.get_operator()
        if op.flags.update_output:
            op.direct(x, out, operation=operation)
        else:
            op.direct(x, out)

    def reshapein(self, shape):
        return self.common[1].reshapein(shape)

    def reshapeout(self, shape):
        return self.common[1].reshapeout(shape)

    def toshapein(self, x):
        return self.common[1].toshapein(x)

    def toshapeout(self, x):
        return self.common[1].toshapeout(x)

    def validatein(self, shape):
        self.common[1].validatein(shape)

    def validateout(self, shape):
        self.common[1].validateout(shape)

    def __repr__(self):
        number = self.number
        cls = self.common[1].__name__
        source = '\n'.join(inspect.getsource(self.callback).split('\n')[:2])
        if self.shapein is not None:
            sin = ', shapein={0}'.format(strshape(self.shapein))
        else:
            sin = ''
        if sin:
            sout = ', shapeout={0}'.format(strshape(self.shapeout))
        else:
            sout = ''
        return '{0}({1}, {2}, {3!r}{4}{5})'.format(
            type(self).__name__, number, cls, source, sin, sout)

    __str__ = __repr__


class ProxyReverseBase(ProxyBase):
    def reshapein(self, shape):
        return self.common[1].reshapeout(shape)

    def reshapeout(self, shape):
        return self.common[1].reshapein(shape)

    def toshapein(self, x):
        return self.common[1].toshapeout(x)

    def toshapeout(self, x):
        return self.common[1].toshapein(x)

    def validatein(self, shape):
        self.common[1].validateout(shape)

    def validateout(self, shape):
        self.common[1].validatein(shape)


class ProxyOperator(ProxyBase):
    """
    Proxy operators, for on-the-fly computations.

    This operator is meant to be used in a group of proxy operators. When
    a member of such a group is called, a callback function is used to access
    the actual operator. This operator is then cached and subsequent uses of
    this operator (including the associated operators, such as conjugate,
    transpose, etc.) will not require another call to the potentially expensive
    callback function. For example, given the group of proxy operators
    [o1, o2, o3], the sum o1.T * o1 + o2.T * o2 + o3.T * o3 only makes three
    calls to the callback function.

    """
    def __init__(self, number, common, callback, flags, flags_conjugate=None,
                 flags_transpose=None, flags_adjoint=None, flags_inverse=None,
                 flags_inverse_conjugate=None, flags_inverse_transpose=None,
                 flags_inverse_adjoint=None, **keywords):
        ProxyBase.__init__(self, number, common, callback, flags, **keywords)
        self.flags_conjugate = flags_conjugate
        self.flags_transpose = flags_transpose
        self.flags_adjoint = flags_adjoint
        self.flags_inverse = flags_inverse
        self.flags_inverse_conjugate = flags_inverse_conjugate
        self.flags_inverse_transpose = flags_inverse_transpose
        self.flags_inverse_adjoint = flags_inverse_adjoint
        self.set_rule('C', lambda s: ProxyConjugateOperator(
            s.number, s.common, s.callback, s.flags_conjugate))
        self.set_rule('T', lambda s: ProxyTransposeOperator(
            s.number, s.common, s.callback, s.flags_transpose))
        self.set_rule('H', lambda s: ProxyAdjointOperator(
            s.number, s.common, s.callback, s.flags_adjoint))
        self.set_rule('I', lambda s: ProxyInverseOperator(
            s.number, s.common, s.callback, s.flags_inverse))
        self.set_rule('IC', lambda s: ProxyInverseConjugateOperator(
            s.number, s.common, s.callback, s.flags_inverse_conjugate))
        self.set_rule('IT', lambda s: ProxyInverseTransposeOperator(
            s.number, s.common, s.callback, s.flags_inverse_transpose))
        self.set_rule('IH', lambda s: ProxyInverseAdjointOperator(
            s.number, s.common, s.callback, s.flags_inverse_adjoint))

    def __getattr__(self, name):
        return getattr(self.get_operator(), name)


class ProxyConjugateOperator(ProxyBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).C


class ProxyTransposeOperator(ProxyReverseBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).T


class ProxyAdjointOperator(ProxyReverseBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).H


class ProxyInverseOperator(ProxyReverseBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).I


class ProxyInverseConjugateOperator(ProxyReverseBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).I.C


class ProxyInverseTransposeOperator(ProxyBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).I.T


class ProxyInverseAdjointOperator(ProxyBase):
    def get_operator(self):
        return ProxyBase.get_operator(self).I.H


def proxy_group(n, callback, shapeins=None, shapeouts=None):
    """
    Return a group of proxy operators, for on-the-fly computations.

    When a member of such a group is called, a callback function is used
    to access the actual operator. This operator is then cached and subsequent
    uses of this operator (including the associated operators, such as
    conjugate, transpose, etc.) will not require another call to the
    potentially expensive callback function. In a proxy group, only one
    instance of actual operator is kept in an object that is common to all
    members of the group. For example, given the group of proxy operators
    [o1, o2, o3], the sum o1.T * o1 + o2.T * o2 + o3.T * o3 only calls three
    times the callback function.

    Note
    ----
    By default, it is assumed that the proxies have the same input and output
    shape. If it is not the case, all the shapes should be specified with the
    'shapeins' and 'shapeouts' keywords.
    It is also assumed that all the proxies have the same flags.

    Parameters
    ----------
    n : int
        The number of elements in the proxy group.

    callback : function
        A function with a single integer argument that ranges from 0 to n-1.
        Its output is an Operator, and its class and flags should be the same.

    Example
    -------
    >>> import numpy as np
    >>> from pyoperators import BlockColumnOperator, DenseOperator, proxy_group
    >>> NPROXIES = 3
    >>> N = 1000
    >>> counter = 0
    >>> def callback(number):
    ...     global counter
    ...     counter += 1
    ...     np.random.seed(number)
    ...     return DenseOperator(np.random.standard_normal((N, N)))
    >>> group = proxy_group(NPROXIES, callback)
    >>> op = BlockColumnOperator(group, new_axisout=0)
    >>> opTop = op.T * op
    >>> y = opTop(np.ones(N))
    >>> print(counter)
    3

    """
    op = callback(0)
    flags = op.flags
    flags_c = op.C.flags
    flags_t = op.T.flags
    flags_h = op.H.flags
    flags_i = op.I.flags
    flags_ic = op.I.C.flags
    flags_it = op.I.T.flags
    flags_ih = op.I.H.flags
    if shapeins is None:
        shapeins = n * (op.shapein,)
    if shapeouts is None:
        shapeouts = n * (op.shapeout,)

    common = [0, op]
    ops = [ProxyOperator(i, common, callback, flags, dtype=op.dtype,
                         shapein=si, shapeout=so,
                         flags_conjugate=flags_c,
                         flags_transpose=flags_t,
                         flags_adjoint=flags_h,
                         flags_inverse=flags_i,
                         flags_inverse_conjugate=flags_ic,
                         flags_inverse_transpose=flags_it,
                         flags_inverse_adjoint=flags_ih)
           for i, (si, so) in enumerate(zip(shapeins, shapeouts))]
    return ops
