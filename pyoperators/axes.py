from __future__ import division

import numpy as np
from pyoperators import CompositionOperator, IdentityOperator, Operator
from pyoperators.utils import ifirst_is_not, product, strshape, tointtuple
from pyoperators.decorators import contiguous, involutary, inplace_only, orthogonal


@orthogonal
@inplace_only
class AxisPermutationOperator(Operator):
    def __init__(self, axes, **keywords):
        axes = tuple(tuple(a) if a is not None else None for a in axes)
        ndmin = ifirst_is_not(axes, None)
        if all(
            a == tuple(r) for a, r in zip(axes[ndmin:], self._identity_axes()[ndmin:])
        ):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        Operator.__init__(self, **keywords)
        self.axes = axes
        self.set_rule(
            'I',
            lambda s: AxisPermutationOperator(
                [np.argsort(a) if a is not None else None for a in s.axes]
            ),
        )
        self.set_rule(
            ('.', AxisPermutationOperator), self._rule_composition, CompositionOperator
        )

    def restridein(self, strides, shape, dtype):
        axes = self.axes[len(strides)]
        return tuple(strides[a] for a in axes)

    def restrideout(self, strides, shape, dtype):
        axes = np.argsort(self.axes[len(strides)])
        return tuple(strides[a] for a in axes)

    def direct(self, input, output):
        pass

    @staticmethod
    def _rule_composition(self, other):
        ndmin = max(ifirst_is_not(self.axes, None), ifirst_is_not(other.axes, None))
        new_axes = ndmin * [None] + [
            [self.axes[i][other.axes[i][j]] for j in range(i)] for i in range(ndmin, 32)
        ]
        return AxisPermutationOperator(new_axes)

    @staticmethod
    def _identity_axes():
        return [None] + [list(range(n)) for n in range(1, 32)]


class AxisRollOperator(AxisPermutationOperator):
    def __init__(self, axis, start=0, **keywords):
        axis = int(axis)
        start = int(start)
        ndmin = max(
            axis + 1 if axis >= 0 else -axis, start + 1 if start >= 0 else -start
        )
        axes = self._identity_axes()
        axes[:ndmin] = ndmin * (None,)
        for n in range(ndmin, 32):
            axis_ = axis if axis >= 0 else n + axis
            start_ = start if start >= 0 else n + start
            if start_ >= axis_:
                start_ -= 1
            a = axes[n][:axis_] + axes[n][axis_ + 1 :]
            a.insert(start_, axes[n][axis_])
            axes[n] = a
        self.axis = axis
        self.start = start
        AxisPermutationOperator.__init__(self, axes, **keywords)


@involutary
class AxisSwapOperator(AxisPermutationOperator):
    def __init__(self, axis1, axis2, **keywords):
        axis1 = int(axis1)
        axis2 = int(axis2)
        axis1, axis2 = min(axis1, axis2), max(axis1, axis2)
        if axis1 < 0:
            axis1, axis2 = axis2, axis1
        axes = self._identity_axes()
        ndmin = max(
            axis1 + 1 if axis1 >= 0 else -axis1, axis2 + 1 if axis2 >= 0 else -axis2
        )
        axes[:ndmin] = ndmin * (None,)
        for n in range(ndmin, 32):
            axis1_ = axis1 if axis1 >= 0 else n + axis1
            axis2_ = axis2 if axis2 >= 0 else n + axis2
            axes[n][axis1_] = axis2_
            axes[n][axis2_] = axis1_
        self.axis1 = axis1
        self.axis2 = axis2
        AxisPermutationOperator.__init__(self, axes, **keywords)


@involutary
class AxisTransposeOperator(AxisPermutationOperator):
    def __init__(self, **keywords):
        axes = [None] + [reversed(a) for a in self._identity_axes()[1:]]
        AxisPermutationOperator.__init__(self, axes, **keywords)


@orthogonal
@contiguous
@inplace_only
class ReshapeOperator(Operator):
    """
    Operator that reshapes arrays.

    Example
    -------
    >>> op = ReshapeOperator(6, (3,2))
    >>> op(np.ones(6)).shape
    (3, 2)
    """

    def __init__(self, shapein, shapeout, **keywords):
        if shapein is None:
            raise ValueError('The input shape is None.')
        if shapeout is None:
            raise ValueError('The output shape is None.')
        shapein = tointtuple(shapein)
        shapeout = tointtuple(shapeout)
        if shapein == shapeout:
            self.__class__ = IdentityOperator
            self.__init__(shapein=shapein, **keywords)
            return
        if product(shapein) != product(shapeout):
            raise ValueError('The total size of the output must be unchanged.')
        Operator.__init__(self, shapein=shapein, shapeout=shapeout, **keywords)
        self.set_rule('I', lambda s: ReshapeOperator(s.shapeout, s.shapein))
        self.set_rule(
            ('.', ReshapeOperator), self._rule_composition, CompositionOperator
        )

    def direct(self, input, output):
        pass

    @staticmethod
    def _rule_composition(self, other):
        return ReshapeOperator(self.shapeout, other.shapein)

    def __str__(self):
        return strshape(self.shapeout) + '←' + strshape(self.shapein)


def strides_contiguous(shape, dtype):
    return tuple(np.cumproduct((dtype.itemsize,) + shape[:0:-1])[::-1])


def iscontiguous(dtype, shape, strides):
    return strides_contiguous(dtype, shape) == strides


def _analyse_strides(
    ops,
):
    """
    Analyse the strides requirement of a group of operators.

    """
    pass


tr = AxisTransposeOperator()
op = tr * tr
assert isinstance(op, IdentityOperator)

sw = AxisSwapOperator(0, 4)
op = sw * sw
assert isinstance(op, IdentityOperator)
