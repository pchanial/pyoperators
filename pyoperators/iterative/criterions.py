"""
This module is obsolete.
It implements Criterions. Those are functions defined from
Norms and Operators to be minimized by iterative algorithms (such as
the conjugate-gradient).

Interfaces with the scipy.optimize algorithms are defined through
their __call__ and diff methods and their shapein attribute.
"""

import copy
import numpy as np
from ..core import Operator, IdentityOperator

__all__ = ['norm2',
           'dnorm2',
           'normp',
           'dnormp',
           'huber',
           'dhuber',
           'hnorm',
           'dhnorm',
           'Norm2',
           'Huber',
           'Normp',
           'Criterion',
           'quadratic_criterion',
           'huber_criterion',
           'normp_criterion']

# norms
# =====

# 2-norm
def norm2(x):
    return np.dot(x.ravel().T, x.ravel())

def dnorm2(x):
    return 2 * x

# p-norm
def normp(p=2):
    def norm(t):
        return np.sum(np.abs(t) ** p)
    return norm

def dnormp(p=2):
    def norm(t):
        return np.sign(t) * p * (np.abs(t) ** (p - 1))
    return norm

# huber norm
def huber(t, delta=1):
    """Apply the huber function to the vector t, with transition delta"""
    t_out = t.flatten()
    quadratic_index = np.where(np.abs(t_out) < delta)
    linear_index = np.where(np.abs(t_out) >= delta)
    t_out[quadratic_index] = np.abs(t_out[quadratic_index]) ** 2
    t_out[linear_index] = 2 * delta * np.abs(t_out[linear_index]) - delta ** 2
    return np.reshape(t_out, t.shape)

def dhuber(t, delta=1):
    """Apply the derivation of the Huber function to t, transition: delta"""
    t_out = t.flatten()
    quadratic_index = np.where(np.abs(t_out) < delta)
    linear_index_positive = np.where(t_out >= delta)
    linear_index_negative = np.where(t_out <= - delta)
    t_out[quadratic_index] = 2 * t_out[quadratic_index]
    t_out[linear_index_positive] = 2 * delta
    t_out[linear_index_negative] = - 2 * delta
    return np.reshape(t_out, t.shape)

def hnorm(d=None):
    if d is None:
        return norm2
    else:
        def norm(t):
            return np.sum(huber(t, d))
        return norm

def dhnorm(d=None):
    if d is None:
        return dnorm2
    else:
        def norm(t):
            return dhuber(t, d)
        return norm

# for operations on norms
def _scalar_mul(func1, scalar):
    def func(x):
        return scalar * func1(x)
    return func

# norm classes
class Norm(object):
    """
    An abstract class to define norm classes.
    """
    def __call__(self, x):
        return self._call(x)
    def diff(self, x):
        return self._diff(x)
    def __mul__(self, x):
        # returns a norm with modified _call and _diff
        if np.isscalar(x):
            kwargs = dict((k,v) for k,v in self.__dict__.items() \
                          if k[0] != '_')
            N = type(self)(kwargs)
            N._call = _scalar_mul(self._call, x)
            N._diff = _scalar_mul(self._diff, x)
            if hasattr(N, "_hessian"):
                N._hessian = _scalar_mul(self._hessian, x)
        else:
            raise ValueError("Expects only scalar multiplication")
        return N
    __imul__ = __mul__
    __rmul__ = __mul__

class Norm2(Norm):
    """
    A norm-2 class. Optionally accepts a covariance matrix C.
    If C is given, the norm would be : np.dot(x.T, C * x).
    Otherwise, it would be norm2(x).

    Parameters
    ----------

    C : LinearOperator (None)
        The covariance matrix of the norm.

    Returns
    -------
    Returns a Norm2 instance with a __call__ and a diff method.
    """
    def __init__(self, C=None):
        def call(x):
            return norm2(x)
        def diff(x):
            return 2 * x
        def hessian(x):
            return 2 * IdentityOperator(shapein=x.size)
        def c_call(x):
            return np.dot(x.T, C * x)
        def c_diff(x):
            return 2 * C * x
        def c_hessian(x):
            return 2 * C
        self.C = C
        if C is None:
            self._call = call
            self._diff = diff
            self._hessian = hessian
        else:
            self._call = c_call
            self._diff = c_diff
            self._hessian = c_hessian

class Huber(Norm):
    """
    An Huber norm class.

    Parameters
    ----------

    delta: float
       The Huber parameter of the norm.
       if abs(x_i) is below delta, returns x_i ** 2
       else returns 2 * delta * x_i - delta ** 2

    Returns
    -------
    Returns an Huber instance with a __call__ and a diff method.
     """
    def __init__(self, delta):
        self.delta = delta
        self._call = hnorm(d=delta)
        self._diff = dhnorm(d=delta)

class Normp(Norm):
    """
    An Norm-p class.

    Parameters
    ----------

    p: float
       The power of the norm.
       The norm will be np.sum(np.abs(x) ** p)

    Returns
    -------
    Returns a Normp instance with a __call__ and a diff method.
     """
    def __init__(self, p):
        self.p = p
        self._call = normp(p=p)
        self._diff = dnormp(p=p)

# criterion elements
# ==================

class CriterionElement(object):
    def __init__(self, norm, op, data=None):
        # test inputs
        if not isinstance(norm, Norm):
            raise ValueError("First parameter should be a Norm instance")
        self.norm = norm
        if not isinstance(op, Operator):
            raise ValueError("First parameter should be an Operator instance")
        self.op = op
        self.shapein = op.shapein
        if not (isinstance(data, np.ndarray) or data is None):
            raise ValueError("data parameter should be ndarray or None")
        if data is not None and not data.shape == np.prod(op.shapeout):
            raise ValueError("data shape sould equal operator shapeout")
        self.data = data

        # cache result
        self.last_x = None
        self.last_ox = None

        # define call and diff
        def _call(x):
            if not self._islastx(x):
                self._storex(x)
            return self.norm(self.last_ox)
        def _diff(x):
            if not self._islastx(x):
                self._storex(x)
            return self.op.T * self.norm.diff(self.last_ox)
        def _data_call(x):
            if not self._islastx(x):
                self._storex(x)
            return self.norm(self.last_ox - data)
        def _data_diff(x):
            if not self._islastx(x):
                self._storex(x)
            return self.op.T * self.norm.diff(self.last_ox - data)
        if data is None:
            self._call = _call
            self._diff = _diff
        else:
            self._call = _data_call
            self._diff = _data_diff

    def _islastx(self, x):
        return np.all(x == self.last_x)

    def _storex(self, x):
        self.last_x = copy.copy(x)
        self.last_ox = self.op * x

    def __call__(self, x):
        return self._call(x)

    def diff(self, x):
        return self._diff(x)

    def __mul__(self, x):
        """returns a criterion element with modified norm"""
        if np.isscalar(x):
            new_norm = x * self.norm
            return CriterionElement(new_norm, self.op, self.data)
        else:
            raise ValueError("Expects only scalar multiplication")
    __imul__ = __mul__
    __rmul__ = __mul__

    def __add__(self, x):
        """Returns a criterion"""
        if isinstance(x, CriterionElement):
            if self.shapein != x.shapein:
                raise ValueError("CriterionElements should have the same shape.")
            return Criterion([self, x])
        elif isinstance(x, Criterion):
            if self.shapein != x.shapein:
                raise ValueError("CriterionElements should have the same shape.")
            return Criterion([self, ] + x.elements)
        elif x == 0.:
            return Criterion([self,])
        else:
            raise ValueError("Expects Criterion or CriterionElement")
    __radd__ = __add__
    __iadd__ = __add__

# criterions
# ===========

class Criterion(object):
    def __init__(self, elements):
        if np.any([el.shapein != elements[0].shapein for el in elements]):
            raise ValueError("CriterionElements should have the same shape.")
        self.elements = elements
        self.shapein = elements[0].shapein
    def __call__(self, x):
        return sum([el(x) for el in self.elements])
    def diff(self, x):
        return sum([el.diff(x) for el in self.elements])

    def __mul__(self, x):
        """returns a criterion element with modified norm"""
        if np.isscalar(x):
            return Criterion([x * e for e in self.elements])
        else:
            raise ValueError("Expects only scalar multiplication")
    __imul__ = __mul__
    __rmul__ = __mul__

    def __add__(self, x):
        """Returns a criterion"""
        if isinstance(x, Criterion):
            return Criterion(self.elements + x.elements)
        elif isinstance(x, CriterionElement):
            return Criterion(self.elements + [x,])
        elif x == 0.:
            return Criterion([self,])
        else:
            raise ValueError("Expects Criterion or scalar")
    __radd__ = __add__
    __iadd__ = __add__

# generate criterions
def quadratic_criterion(model, data, hypers=[], priors=[], covariances=None):
    if covariances is None:
        norms = [Norm2(), ] * (1 + len(hypers))
    else:
        norms = [Norm2(C) for C in covariances]
    likelihood = CriterionElement(norms[0], model, data=data)
    prior_elements = [CriterionElement(n, p) for n, p in zip(norms[1:], priors)]
    prior = sum([h * p for h, p in zip(hypers, prior_elements)])
    criterion = likelihood + prior
    return criterion

def huber_criterion(model, data, hypers=[], priors=[], deltas=[]):
    norms = [Huber(d) for d in deltas]
    likelihood = CriterionElement(norms[0], model, data=data)
    prior_elements = [CriterionElement(n, p) for n, p in zip(norms[1:], priors)]
    prior = sum([h *p for h, p in zip(hypers, prior_elements)])
    criterion = likelihood + prior
    return criterion

def normp_criterion(model, data, hypers=[], priors=[], ps=[]):
    norms = [Normp(p) for p in ps]
    likelihood = CriterionElement(norms[0], model, data=data)
    prior_elements = [CriterionElement(n, p) for n, p in zip(norms[1:], priors)]
    prior = sum([h *p for h, p in zip(hypers, prior_elements)])
    criterion = likelihood + prior
    return criterion
