"""
Module defining stop conditions for iterative algorithms.

"""

from __future__ import absolute_import, division, print_function

__all__ = ['StopCondition',
           'MaxErrorStopCondition',
           'MaxIterationStopCondition']


class StopCondition(object):
    """
    A class defining stop conditions for iterative algorithms. It must be
    called with an Algorithm instance as argument. To stop the iterations,
    the instance must raise a StopIteration instance.

    """
    def __init__(self, condition, message):
        self.condition = condition
        self.message = message

    def __call__(self, s):
        if self.condition(s):
            raise StopIteration(self.message)

    def __or__(self, other):
        return OrStopCondition([self, other])

    def __str__(self):
        return self.message


class OrStopCondition(StopCondition):
    def __init__(self, stop_conditions):
        self.operands = tuple(stop_conditions)

    def __call__(self, s):
        for c in self.operands:
            c(s)

    def __str__(self):
        ' or '.join(str(c) for c in self.operands)


class NoStopCondition(StopCondition):
    def __init__(self):
        StopCondition.__init__(self, lambda s: False, 'no stop condition')

    def __or__(self, other):
        return other


class MaxErrorStopCondition(StopCondition):
    """
    Stop if the 'error' attribute is less than the specified maximum tolerance.

    """
    def __init__(self, maxerror, message='The maximum error is reached.'):
        self.maxerror = maxerror
        StopCondition.__init__(self, lambda s: s.error <= maxerror, message)

    def __str__(self):
        return 'maxerror={0}'.format(self.maxerror)


class MaxIterationStopCondition(StopCondition):
    """
    Stop if the 'niterations' attribute is equal to the specified maximum
    number of iterations.

    """
    def __init__(self, maxiteration, message='The maximum number of iterations'
                 ' is reached.'):
        self.maxiteration = maxiteration
        StopCondition.__init__(self, lambda s: s.niterations == maxiteration,
                               message)

    def __str__(self):
        return 'maxiteration={0}'.format(self.maxiteration)
