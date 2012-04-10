import numexpr
if numexpr.__version__ < 2.0:
    raise ImportError('Please update numexpr to a newer version > 2.0.')

import numpy as np
from .decorators import square, inplace, universal
from .core import Operator
from .utils import operation_assignment, operation_symbol

__all__ = ['ClipOperator',
           'MaximumOperator',
           'MinimumOperator',
           'NumexprOperator',
           'RoundOperator']

@square
@inplace
@universal
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

@square
@inplace
@universal
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
@universal
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


# numexpr 2.0 cannot handle inplace reductions
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
        Operator.__init__(self, dtype=dtype, flags={'inplace_reduction':False},
                          **keywords)

    def direct(self, input, output, operation=operation_assignment):
        if operation is operation_assignment:
            expr = self.expr
        else:
            op = operation_symbol[operation]
            expr = 'output' + op + '(' + self.expr + ')'
        numexpr.evaluate(expr, global_dict=self.global_dict, out=output)


@square
@inplace
@universal
class RoundOperator(Operator):
    """Rounding operator.
    
    The rounding method may be one of the following:
        - rtz : round towards zero (truncation)
        - rti : round towards infinity (Not implemented)
        - rtmi : round towards minus infinity (floor)
        - rtpi : round towards positive infinity (ceil)
        - rhtz : round half towards zero (Not implemented)
        - rhti : round half towards infinity (numpy's round, fortran's nint)
        - rhtmi : round half towards minus infinity (Not implemented)
        - rhtpi : round half towards positive infinity (Not implemented)
        - rhte : round half to even
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
                 #'rhtmi'
                 #'rhtpi'
                 'rhte'  : lambda i,o: np.round(i,0,o),
                 #'rhs'
                 }
        if method not in table:
            raise ValueError('The rounding method must be one of the following:'
                ' ' + ','.join("'" + k + "'" for k in table.keys()) + '.')
        Operator.__init__(self, table[method], **keywords)
        self.__name__ += ' [' + method + ']'

