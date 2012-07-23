import numexpr
if numexpr.__version__ < 2.0:
    raise ImportError('Please update numexpr to a newer version > 2.0.')

import numpy as np
from .decorators import real, square, idempotent, inplace, separable
from .core import (Operator, BlockColumnOperator, CompositionOperator,
                   IdentityOperator, ReductionOperator)
from .utils import operation_assignment, operation_symbol, strenum
from .utils.ufuncs import hard_thresholding, soft_thresholding

__all__ = ['ClipOperator',
           'HardThresholdingOperator',
           'MaxOperator',
           'MinOperator',
           'MinMaxOperator',
           'MaximumOperator',
           'MinimumOperator',
           'NumexprOperator',
           'ProductOperator',
           'RoundOperator',
           'SoftThresholdingOperator']

@square
@inplace
@separable
class ClipOperator(Operator):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Arguments
    ---------
    vmin: scalar or array_like
        The minimum limit below which all input values are set to vmin.
    vmax: scalar or array_like
        The maximum limit above which all input values are set to vmax.

    Exemples
    --------
    >>>  C = ClipOperator(0, 1)
    >>> x = linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> C(x)
    array([ 0.,  0.,  0.,  1.,  1.])

    See also
    --------
    MaximumOperator, MinimumOperator, np.clip

    """
    def __init__(self, vmin, vmax, **keywords):
        Operator.__init__(self, lambda i,o: np.clip(i, vmin, vmax, out=o),
                          **keywords)


class ProductOperator(ReductionOperator):
    """
    Product-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = ProductOperator()
    >>> op([1,2,3])
    array(6)

    """
    def __init__(self, axis=None, dtype=None, skipna=True, **keywords):
        ReductionOperator.__init__(self, np.multiply, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)


class MaxOperator(ReductionOperator):
    """
    Max-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = MaxOperator()
    >>> op([1,2,3])
    array(3)

    """
    def __init__(self, axis=None, dtype=None, skipna=False, **keywords):
        if np.__version__ < '1.8':
            func = np.nanmax if skipna else np.max
        else:
            func = np.max
        ReductionOperator.__init__(self, func, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)


class MinOperator(ReductionOperator):
    """
    Min-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = MinOperator()
    >>> op([1,2,3])
    array(1)

    """
    def __init__(self, axis=None, dtype=None, skipna=False, **keywords):
        if np.__version__ < '1.8':
            func = np.nanmin if skipna else np.min
        else:
            func = np.min
        ReductionOperator.__init__(self, func, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)


class MinMaxOperator(BlockColumnOperator):
    """
    MinMax-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    new_axisout : integer, optional
        Axis in which the minimum and maximum values are set.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = MinMaxOperator()
    >>> op([1,2,3])
    array([1, 3])
    >>> op = MinMaxOperator(axis=0, new_axisout=0)
    >>> op([[1,2,3],[2,1,4],[0,1,8]])
    array([[0, 1, 3],
           [2, 2, 8]])

    """
    def __init__(self, axis=None, dtype=None, skipna=False, new_axisout=-1,
                 **keywords):
        operands = [MinOperator(axis=axis, dtype=dtype, skipna=skipna),
                    MaxOperator(axis=axis, dtype=dtype, skipna=skipna)]
        BlockColumnOperator.__init__(self, operands, new_axisout=new_axisout,
                                     **keywords)


@square
@inplace
@separable
class MaximumOperator(Operator):
    """
    Set all input array values above a given value to this value.

    Arguments
    ---------
    value: scalar or array_like
        Threshold value to which the input array is compared.

    Exemple
    -------
    >>> M = MaximumOperator(1)
    >>> x =  linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> M(x)
    array([ 1.,  1.,  1.,  1.,  2.])

    See also
    --------
    ClipOperator, MinimumOperator, np.maximum

    """
    def __init__(self, value, **keywords):
        Operator.__init__(self, lambda i,o: np.maximum(i, value, o), **keywords)


@square 
@inplace
@separable
class MinimumOperator(Operator):
    """
    Set all input array values above a given value to this value.

    Arguments
    ---------
    value: scalar, broadcastable array
        The value to which the input array is compared.

    Exemple
    -------
    >>> M = MinimumOperator(1)
    >>> x =  linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> M(x)
    array([-2., -1.,  0.,  1.,  1.])

    See also
    --------
    ClipOperator, MaximumOperator, np.minimum

    """
    def __init__(self, value, **keywords):
        Operator.__init__(self, lambda i,o: np.minimum(i, value, o), **keywords)


@square
@inplace
class NumexprOperator(Operator):
    """
    Return an operator evaluating an expression using numexpr.
    
    Parameters
    ----------
    expr : string
        The numexp expression to be evaluated. It must contain the 'input'
        variable name.
    global_dict : dict
        A dictionary of global variables that are passed to numexpr's 'evaluate'
        method.

    Example
    -------
    >>> k = 1.2
    >>> op = NumexprOperator('exp(input+k)', {'k':k})
    >>> print op(1) == np.exp(2.2)
    True

    """
    def __init__(self, expr, global_dict=None, dtype=float, **keywords):
        self.expr = expr
        self.global_dict = global_dict
        if numexpr.__version__ < '2.0.2':
            keywords['flags'] = self.validate_flags(keywords.get('flags', {}),
                                                    inplace_reduction=False)
        Operator.__init__(self, dtype=dtype, **keywords)

    def direct(self, input, output, operation=operation_assignment):
        if operation is operation_assignment:
            expr = self.expr
        else:
            op = operation_symbol[operation]
            expr = 'output' + op + '(' + self.expr + ')'
        numexpr.evaluate(expr, global_dict=self.global_dict, out=output)


@square
@idempotent
@inplace
@separable
class RoundOperator(Operator):
    """Rounding operator.
    
    The rounding method may be one of the following:
        - rtz : round towards zero (truncation)
        - rti : round towards infinity (Not implemented)
        - rtmi : round towards minus infinity (floor)
        - rtpi : round towards positive infinity (ceil)
        - rhtz : round half towards zero (Not implemented)
        - rhti : round half towards infinity (Fortran's nint)
        - rhtmi : round half towards minus infinity
        - rhtpi : round half towards positive infinity
        - rhte : round half to even (numpy's round), 
        - rhto : round half to odd
        - rhs : round half stochastically (Not implemented)

    """
    def __init__(self, method='rhte', **keywords):
        method = method.lower()
        table = {'rtz'   : np.trunc,
                 #'rti'
                 'rtmi'  : np.floor,
                 'rtpi'  : np.ceil,
                 #'rhtz'
                 #'rhti'
                 'rhtmi' : self._direct_rhtmi,
                 'rhtpi' : self._direct_rhtpi,
                 'rhte'  : lambda i,o: np.round(i,0,o),
                 #'rhs'
                 }
        if method not in table:
            raise ValueError('Invalid rounding method. Expected values are {0}.'
                             .format(strenum(table.keys())))
        Operator.__init__(self, table[method], **keywords)
        self.method = method

    @staticmethod
    def _direct_rhtmi(input, output):
        """ Round half to -inf. """
        np.add(input, 0.5, output)
        np.ceil(output, output)
        np.add(output, -1, output)

    @staticmethod
    def _direct_rhtpi(input, output):
        """ Round half to +inf. """
        np.add(input, -0.5, output)
        np.floor(output, output)
        np.add(output, 1, output)


@real
@square
@idempotent
@inplace
@separable
class HardThresholdingOperator(Operator):
    """
    Hard thresholding operator.

    Ha(x) = x if |x| > a,
            0 otherwise.

    Parameter
    ---------
    a : positive float or array
        The hard threshold.

    """
    def __init__(self, a, **keywords):
        a = np.asarray(a)
        if np.any(a < 0):
            raise ValueError('Negative hard threshold.')
        if a.ndim > 0:
            keywords['shapein'] = a.shape
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        if np.all(a == 0):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        Operator.__init__(self, **keywords)
        self.a = a
        self.set_rule('.{HardThresholdingOperator}', lambda s, o:
                      HardThresholdingOperator(np.maximum(s.a, o.a)),
                      CompositionOperator)

    def direct(self, input, output):
        hard_thresholding(input, self.a, output)


@real
@square
@inplace
@separable
class SoftThresholdingOperator(Operator):
    """
    Soft thresholding operator.

    Sa(x) = sign(x) [|x| - a]+

    Parameter
    ---------
    a : positive float or array
        The soft threshold.

    """
    def __init__(self, a, **keywords):
        a = np.asarray(a)
        if np.any(a < 0):
            raise ValueError('Negative soft threshold.')
        if a.ndim > 0:
            keywords['shapein'] = a.shape
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        if np.all(a == 0):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        Operator.__init__(self, **keywords)
        self.a = a

    def direct(self, input, output):
        soft_thresholding(input, self.a, output)
