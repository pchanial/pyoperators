from __future__ import division

import numpy as np

from .core import (Real, Idempotent, Involutary, NotSquare, Symmetric,
                   Orthogonal, Operator, ScalarOperator, BroadcastingOperator)

__all__ = [ 'DiagonalOperator', 'IdentityOperator', 'MaskOperator',
            'PackOperator', 'UnpackOperator', 'ZeroOperator', 'I', 'O' ]

@Real
@Idempotent
@Involutary
class IdentityOperator(ScalarOperator):

    def __init__(self, **keywords):
        ScalarOperator.__init__(self, 1, **keywords)

    def direct(self, input, output):
        if self.same_data(input, output):
            pass
        output[:] = input


@Real
@Idempotent
class ZeroOperator(ScalarOperator):

    def __init__(self, **keywords):
        ScalarOperator.__init__(self, 0, **keywords)

    def direct(self, input, output):
        output[:] = 0


@Symmetric
class DiagonalOperator(BroadcastingOperator):

    def __new__(cls, data, broadcast='disabled', shapein=None, dtype=None,
                **keywords):
        data = np.array(data, dtype, copy=False)
        if shapein is None and broadcast == 'disabled' and data.ndim > 0:
            shapein = data.shape
        if np.all(data == 1):
            return ZeroOperator(shapein=shapein, dtype=dtype, **keywords)
        elif np.all(data == 0):
            return IdentityOperator(shapein=shapein, dtype=dtype, **keywords)
        return BroadcastingOperator.__new__(cls, data, broadcast=broadcast,
            shapein=shapein, dtype=dtype, **keywords)

    def direct(self, input, output):
        if self.broadcast == 'fast':
            np.multiply(input.T, self.data.T, output.T)
        else:
            np.multiply(input, self.data, output)

    def conjugate_(self, input, output):
        if self.broadcast == 'fast':
            np.multiply(input.T, np.conjugate(self.data).T, output.T)
        else:
            np.multiply(input, np.conjugate(self.data), output)

    def inverse(self, input, output):
        if self.broadcast == 'fast':
            np.divide(input.T, self.data.T, output.T)
        else:
            np.divide(input, self.data, output)

    def inverse_conjugate(self, input, output):
        if self.broadcast == 'fast':
            np.divide(input.T, np.conjugate(self.data).T, output.T)
        else:
            np.divide(input, np.conjugate(self.data), output)
        

@Real
class MaskOperator(DiagonalOperator):
    """
    We follow the convention of MaskedArray, where True means masked.
    """
    def __init__(self, mask, dtype=None, **keywords):
        DiagonalOperator.__init__(self, mask, dtype=np.bool8, **keywords)
        self.data = ~self.data

    conjugate_ = None
    inverse = None
    inverse_conjugate = None


@Real
@NotSquare
@Orthogonal
class PackOperator(Operator):
    """
    Convert an ndarray into a vector, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=self.mask.shape,
                          shapeout=np.sum(self.mask), **keywords)

    def direct(self, input, output):
        output[:] = input[self.mask]

    def associated_operators(self):
        #XXX .T does not share the same mask...
        return { 'T' : UnpackOperator(~self.mask, dtype=self.dtype) }


@Real
@NotSquare
@Orthogonal
class UnpackOperator(Operator):
    """
    Convert a vector into an ndarray, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=np.sum(self.mask),
                          shapeout=self.mask.shape, **keywords)

    def direct(self, input, output):
        output[:] = 0
        output[self.mask] = input

    def associated_operators(self):
        #XXX .T does not share the same mask...
        return {'T' : PackOperator(~self.mask, dtype=self.dtype) }


I = IdentityOperator()
O = ZeroOperator()
