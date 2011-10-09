import numpy as np

from .decorators import square, inplace
from .core import Operator

__all__ = ['ClipOperator', 'MaximumOperator', 'MinimumOperator', 'RoundOperator']


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
        Operator.__init__(self, lambda i, o: np.clip(i, vmin, vmax, out=o), **keywords)


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
        Operator.__init__(self, lambda i, o: np.maximum(i, value, o), **keywords)


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
        Operator.__init__(self, lambda i, o: np.minimum(i, value, o), **keywords)


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
        table = {
            'rtz': np.trunc,
            #'rti'
            'rtmi': np.floor,
            'rtpi': np.ceil,
            #'rhtz'
            #'rhti'
            #'rhtmi'
            #'rhtpi'
            'rhte': lambda i, o: np.round(i, 0, o),
            #'rhs'
        }
        if method not in table:
            raise ValueError(
                'The rounding method must be one of the following:'
                ' ' + ','.join("'" + k + "'" for k in table.keys()) + '.'
            )
        Operator.__init__(self, table[method], **keywords)
        self.__name__ += ' [' + method + ']'
