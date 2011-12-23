try:
    import numexpr
except ImportError:
    numexpr = None

import numpy as np
from .decorators import square, inplace
from .core import Operator

__all__ = ['ClipOperator', 'MaximumOperator', 'MinimumOperator',
           'RoundOperator']
if numexpr:
    __all__ += ['NumexprOperator']

@square
@inplace
class ClipOperator(Operator):
    def __init__(self, vmin, vmax, **keywords):
        """
        Clips the values of the input arrays.

        Arguments
        ---------

        vmin: scalar
          The minimum limit below which all input values are set to vmin.
        vmax: scalar
          The maximum limit above which all input values are set to vmax.

        Exemples
        --------
        >>> C = ClipOperator(0, 1, shapein=5)
        >>> x =  linspace(-2, 2, 5)
        >>> x
        array([-2., -1.,  0.,  1.,  2.])
        >>> C * x
        Info: Allocating (5,) float64 = 40 bytes in ClipOperator.
        ndarraywrap([ 0.,  0.,  0.,  1.,  1.])

        See also
        --------
        np.clip
        """
        Operator.__init__(self, lambda i,o: np.clip(i, vmin, vmax, out=o),
                          **keywords)

@square
@inplace
class MaximumOperator(Operator):
    def __init__(self, value, **keywords):
        """
        Set all input array values below value to value.

        Arguments
        ---------
        value: scalar
          The value with which the input array is compared.

        Exemple
        -------
        >>> M = MaximumOperator(1, shapein=5)
        >>> x =  linspace(-2, 2, 5)
        >>> x
        array([-2., -1.,  0.,  1.,  2.])
        >>> M * x
        Info: Allocating (5,) float64 = 40 bytes in MaximumOperator.
        ndarraywrap([ 1.,  1.,  1.,  1.,  2.])

        See also
        --------
        np.maximum
        """
        Operator.__init__(self, lambda i,o: np.maximum(i, value, o), **keywords)


@square 
@inplace
class MinimumOperator(Operator):
    def __init__(self, value, **keywords):
        """
        Set all input array values above value to value.

        Arguments
        ---------
        value: scalar
          The value with which the input array is compared.

        Exemple
        -------
        >>> M = MinimumOperator(1, shapein=5)
        >>> x =  linspace(-2, 2, 5)
        >>> x
        array([-2., -1.,  0.,  1.,  2.])
        >>> M * x
        Info: Allocating (5,) float64 = 40 bytes in MinimumOperator.
        ndarraywrap([-2., -1.,  0.,  1.,  1.])

        See also
        --------
        np.minimum
        """
        Operator.__init__(self, lambda i,o: np.minimum(i, value, o), **keywords)


if numexpr:
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
        Operator.__init__(self, dtype=dtype, **keywords)

    if numexpr.__version__ >= 2.0:
      def direct(self, input, output):
        numexpr.evaluate(self.expr, global_dict=self.global_dict, out=output)
    else:
      def direct(self, input, output):
        output[...] = numexpr.evaluate(self.expr, global_dict=self.global_dict)


@square
@inplace
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

