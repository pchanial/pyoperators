import numpy as np

from .core import Square, Operator

__all__ = ['ClipOperator', 'MaximumOperator', 'MinimumOperator', 'RoundOperator']


@Square
class ClipOperator(Operator):
    def __init__(self, vmin, vmax, **keywords):
        Operator.__init__(self, lambda i, o: np.clip(i, vmin, vmax, out=o), **keywords)


@Square
class MaximumOperator(Operator):
    def __init__(self, value, **keywords):
        Operator.__init__(self, lambda i, o: np.maximum(i, value, o), **keywords)


@Square
class MinimumOperator(Operator):
    def __init__(self, value, **keywords):
        Operator.__init__(self, lambda i, o: np.minimum(i, value, o), **keywords)


@Square
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
                'The rounding method must be one of the following'
                ': ' + ','.join("'" + k + "'" for k in table.keys()) + '.'
            )
        Operator.__init__(self, table[method], **keywords)
        self.__name__ += ' [' + method + ']'
