from __future__ import division
import numpy as np
import time

from ..core import IdentityOperator, asoperator
from ..memory import empty, zeros
from .core import AbnormalStopIteration, IterativeAlgorithm
from .stopconditions import MaxIterationStopCondition

__all__ = ['pcg']


class PCGAlgorithm(IterativeAlgorithm):
    """
    OpenMP/MPI Preconditioned conjugate gradient iteration to solve A x = b.

    """
    def __init__(self, A, b, x0=None, tol=1.e-5, maxiter=300, M=None,
                 disp=False, callback=None, reuse_initial_state=False):
        """
        Parameters
        ----------
        A : {Operator, sparse matrix, dense matrix}
            The real or complex N-by-N matrix of the linear system
            ``A`` must represent a hermitian, positive definite matrix
        b : {array, matrix}
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x0  : {array, matrix}
            Starting guess for the solution.
        tol : float, optional
            Tolerance to achieve. The algorithm terminates when either the
            relative residual is below `tol`.
        maxiter : integer, optional
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M : {Operator, sparse matrix, dense matrix}, optional
            Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.
        disp : boolean
            Set to True to display convergence message
        callback : function, optional
            User-supplied function to call after each iteration.  It is called
            as callback(self), where self is an instance of this class.
        reuse_initial_state : boolean, optional
            If set to True, the buffer initial guess (if provided) is reused
            during the iterations. Beware of side effects!

        Returns
        -------
        x : array
            The converged solution.

        Raises
        ------
        pyoperators.AbnormalStopIteration : if the solver reached the maximum
            number of iterations without reaching specified tolerance.

        """
        dtype = A.dtype or np.dtype(float)
        if dtype.kind == 'c':
            raise TypeError('The complex case is not yet implemented.')
        elif dtype.kind != 'f':
            dtype = np.dtype(float)
        b = np.array(b, dtype, copy=False)

        if x0 is None:
            x0 = zeros(b.shape, dtype)

        abnormal_stop_condition = MaxIterationStopCondition(
            maxiter, 'Solver reached maximum number of iterations without reac'
            'hing specified tolerance.')

        IterativeAlgorithm.__init__(
            self, x=x0, abnormal_stop_condition=abnormal_stop_condition,
            disp=disp, dtype=dtype, reuse_initial_state=reuse_initial_state,
            inplace_recursion=True, callback=callback)

        A = asoperator(A)
        if A.shapein is None:
            raise ValueError('The operator input shape is not explicit.')
        if A.shapein != b.shape:
            raise ValueError(
                "The operator input shape '{0}' is incompatible with that of t"
                "he RHS '{1}'.".format(A.shapein, b.shape))
        self.A = A
        self.b = b
        self.comm = A.commin
        self.norm = lambda x: _norm2(x, self.comm)
        self.dot = lambda x, y: _dot(x, y, self.comm)

        if M is None:
            M = IdentityOperator()
        self.M = asoperator(M)

        self.tol = tol
        self.b_norm = self.norm(b)
        self.d = empty(b.shape, dtype)
        self.q = empty(b.shape, dtype)
        self.r = empty(b.shape, dtype)
        self.s = empty(b.shape, dtype)

    def initialize(self):
        IterativeAlgorithm.initialize(self)

        if self.b_norm == 0:
            self.x[...] = 0
            raise StopIteration('RHS is zero.')

        self.r[...] = self.b
        self.r -= self.A(self.x)
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')
        self.M(self.r, self.d)
        self.delta = self.dot(self.r, self.d)

    def iteration(self):
        self.A(self.d, self.q)
        alpha = self.delta / self.dot(self.d, self.q)
        self.x += alpha * self.d
        self.r -= alpha * self.q
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')

        self.M(self.r, self.s)
        delta_old = self.delta
        self.delta = self.dot(self.r, self.s)
        beta = self.delta / delta_old
        self.d *= beta
        self.d += self.s

    @staticmethod
    def callback(self):
        if self.disp:
            print '{0:4}: {1}'.format(self.niterations, self.error)


def pcg(A, b, x0=None, tol=1.e-5, maxiter=300, M=None, disp=False,
        callback=None, reuse_initial_state=False):
    """
    Parameters
    ----------
    A : {Operator, sparse matrix, dense matrix}
        The real or complex N-by-N matrix of the linear system
        ``A`` must represent a hermitian, positive definite matrix
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when either the
        relative residual is below `tol`.
    maxiter : integer, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {Operator, sparse matrix, dense matrix}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    disp : boolean
        Set to True to display convergence message
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(self), where self is an instance of this class.
    reuse_initial_state : boolean, optional
        If set to True, the buffer initial guess (if provided) is reused
        during the iterations. Beware of side effects!

    Returns
    -------
    output : dict whose keys are
        'x' : the converged solution.
        'success' : boolean indicating success
        'message' : string indicating cause of failure

    """
    time0 = time.time()
    algo = PCGAlgorithm(A, b, x0=x0, tol=tol, maxiter=maxiter, disp=disp,
                        M=M, callback=callback,
                        reuse_initial_state=reuse_initial_state)
    try:
        output = algo.run()
        success = True
        message = ''
    except AbnormalStopIteration as e:
        output = algo.finalize()
        success = False
        message = str(e)
    return {'x': output,
            'success': success,
            'message': message,
            'nit': algo.niterations,
            'error': algo.error,
            'time': time.time() - time0}


def _norm2(x, comm):
    x = x.ravel()
    n = np.array(np.dot(x, x))
    if comm is not None:
        comm.Allreduce(n, n)
    return n


def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))
    if comm is not None:
        comm.Allreduce(d, d)
    return d
