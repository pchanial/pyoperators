"""
Line searches: find minimum of a multivariate function. 

Optionnaly depends on scipy.optimize for some line searches.

Available:

- optimal step (exact minimum if Criterion is quadratic (only Norm2
  norms))

- Backtracking : starts with optimal steps and reduces step until
  criterion decreases.

if scipy.optimize is in PYTHONPATH:

- LineSearch, LineSearchArmijo, LineSearchWolfe1; LineSearchWolfe2
"""
import numpy as np
from .criterions import Norm2

__all__ = ['optimal_step',
           'Backtracking',
           'default_backtracking']


def optimal_step(algo):
    """
    Finds quadratic optimal step of a criterion.

    Arguments
    ----------

    algo: Algoritm instance with the following attributes:
      current_descent, current_gradient, criterion. The criterion
      attribute should be a Criterion instance with the following
      attributes: model, priors, hypers, norms.

    Returns
    -------
    a: float
      The optimal step.
    """
    # get variables from criterion
    d = algo.current_descent
    g = algo.current_gradient
    norms = [el.norm for el in algo.criterion.elements]
    # replace norms by Norm2 if not a Norm2 instance
    # to handle properly Norm2 with C covariance matrices ...
    norms = [n if isinstance(n, Norm2) else Norm2() for n in norms]
    ops = [el.op for el in algo.criterion.elements]
    # compute quadratic optimal step
    a = -.5 * np.dot(d.T, g)
    a /= np.sum([N(O * d) for N, O in zip(norms, ops)])
    return a


class Backtracking(object):
    def __init__(self, maxiter=10, tau=.5):
        self.maxiter = maxiter
        self.tau = tau

    def __call__(self, algo):
        x = algo.current_solution
        d = algo.current_descent
        a = optimal_step(algo)
        i = 0
        f0 = algo.current_criterion
        fi = 2 * f0
        while (i < self.maxiter) and (fi > f0):
            i += 1
            a *= self.tau
            xi = x + a * d
            fi = algo.criterion(xi)
        return a

default_backtracking = Backtracking()

# if optimize package available wrap line search for use in algorithms
try:
    from scipy.optimize import linesearch
except ImportError:
    pass

if 'linesearch' in locals():
    class LineSearch(object):
        """
        Wraps scipy.optimize.linesearch.line_search
        """
        def __init__(self, args=(), **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.f = None
            self.fprime = None
            self.xk = None
            self.pk = None
            self.gfk = None
            self.old_fval = None
            self.old_old_fval = None
            self.step = None

        def get_values(self, algo):
            self.f = algo.criterion
            self.fprime = algo.gradient
            self.xk = algo.current_solution
            self.pk = algo.current_descent
            self.gfk = algo.current_gradient
            self.old_fval = algo.current_criterion
            self.old_old_fval = algo.last_criterion

        def _line_search(s):
            line_search = linesearch.line_search
            out = line_search(s.f, s.fprime, s.xk, s.pk, gfk=s.gfk,
                              old_fval=s.old_fval,
                              old_old_fval=s.old_old_fval,
                              args=s.args, **s.kwargs)
            s.step = out[0]

        def __call__(self, algo):
            # get values
            self.get_values(algo)
            # perform line search
            self._line_search()
            # if no output given, fallback to optimal step ...
            if self.step is None:
                self.step = optimal_step(algo)
            return self.step


    class LineSearchArmijo(LineSearch):
        """
        Wraps scipy.optimize.linesearch.line_search_armijo.
        """
        def _line_search(s):
            armijo = linesearch.line_search_armijo
            out = armijo(s.f, s.xk, s.pk, s.gfk, s.old_fval, args=s.args,
                         **s.kwargs)
            s.step = out[0]


    class LineSearchWolfe1(LineSearch):
        """
        Wraps scipy.optimize.linesearch.line_search_wolfe1
        """
        def _line_search(s):
            wolfe1 = linesearch.line_search_wolfe1
            out = wolfe1(s.f, s.fprime, s.xk, s.pk, s.gfk, s.old_fval,
                         s.old_old_fval, args=s.args, **s.kwargs)
            s.step = out[0]


    class LineSearchWolfe2(LineSearch):
        """
        Wraps scipy.optimize.linesearch.line_search_wolfe2
        """
        def _line_search(s):
            wolfe2 = linesearch.line_search_wolfe2
            out = wolfe2(s.f, s.fprime, s.xk, s.pk, s.gfk, s.old_fval,
                         s.old_old_fval, args=s.args, **s.kwargs)
            s.step = out[0]
