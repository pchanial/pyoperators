"""
Wraps scipy.optimize.fmin_* algorithms using Criterion instances.
"""
from copy import copy
import numpy as np
import scipy.optimize as opt

class FminWrapper(object):
    """
    Abstract class to generate wrappers around scipy.optimize fmin_*
    functions.

    Parameters
    -----------

    criterion : Criterion
        A criterion function with __call__ and gradient methods.
    x0 : ndarray (None)
        First guess
    args=() : tuple
        Extra arguments for the criterion function
    kwargs : dict
        Parameters of the fmin_function

    fmin function docstring
    ------------------------
    """
    def __init__(self, criterion, x0=None, *args, **kwargs):
        self.criterion = criterion
        self.gradient = getattr(criterion, "gradient", None)
        self.hessian = getattr(criterion, "hessian", None)
        self.hessian_p = getattr(criterion, "hessian_p", None)
        self.shapein = criterion.shapein
        self.args = args
        self.kwargs = kwargs
        self.first_guess(x0)
        # to store solution
        self.current_solution = None
        self.optimizer_output = None
    def first_guess(self, x0=None):
        """
        Sets current_solution attribute to initial value.
        """
        if x0 is None:
            self.current_solution = np.zeros(self.shapein)
        else:
            self.current_solution = copy(x0)

class FminCOBYLA(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_cobyla.__doc__
    def __init__(self, criterion, cons, x0=None, *args, **kwargs):
        self.cons = cons
        FminWrapper.__init__(self, criterion, x0=x0, *args, **kwargs)
    def __call__(self):
        self.first_guess()
        self.current_solution = opt.fmin_cobyla(self.criterion,
                                                self.current_solution,
                                                self.cons,
                                                args=self.args,
                                                **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminPowell(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_powell.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_powell(self.criterion,
                                                self.current_solution,
                                                args=self.args,
                                                **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminCG(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_cg.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_cg(self.criterion,
                                            self.current_solution,
                                            fprime=self.gradient,
                                            args=self.args,
                                            **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminTNC(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_tnc.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_tnc(self.criterion,
                                             self.current_solution,
                                             fprime=self.gradient,
                                             args=self.args,
                                             **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminNCG(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_ncg.__doc__
    def __init__(self, criterion, x0=None, *args, **kwargs):
        super(FminNCG, self).__init__(criterion, x0=x0, *args, **kwargs)
        if self.hessian_p is None:
            raise ValueError("Cannot run FminNCG if the criterion has no hessian.")
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_ncg(self.criterion,
                                             self.current_solution,
                                             fprime=self.gradient,
                                             fhess_p=self.hessian_p,
                                             args=self.args,
                                             **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminLBFGSB(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_l_bfgs_b.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_l_bfgs_b(self.criterion,
                                                  self.current_solution,
                                                  fprime=self.gradient,
                                                  args=self.args,
                                                  **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminSLSQP(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_slsqp.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_slsqp(self.criterion,
                                               self.current_solution,
                                               fprime=self.gradient,
                                               args=self.args,
                                               **self.kwargs)

        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution

class FminBFGS(FminWrapper):
    __doc__ = FminWrapper.__doc__ + opt.fmin_bfgs.__doc__
    def __call__(self):
        self.first_guess()
        self.optimizer_output = opt.fmin_bfgs(self.criterion,
                                              self.current_solution,
                                              fprime=self.gradient,
                                              args=self.args,
                                              **self.kwargs)
        # output depends on kwargs ...
        if isinstance(self.optimizer_output, tuple):
            self.current_solution = self.optimizer_output[0]
        else:
            self.current_solution = self.optimizer_output
        return self.current_solution
