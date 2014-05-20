from __future__ import absolute_import, division, print_function

import numpy as np
from ..core import asoperator
from ..linear import EigendecompositionOperator, TridiagonalOperator
from .core import IterativeAlgorithm
from .stopconditions import MaxIterationStopCondition


class LanczosAlgorithm(IterativeAlgorithm):
    """
    Tridiagonalization Lanczos step and eigendecomposition at exit.

    http://en.wikipedia.org/wiki/Lanczos_algorithm
    """
    def __init__(self, A, v0=None, maxiter=300):
        """
        Use Lanczos algorithm to approximate a linear Operator.

        Parameters
        ----------
        A: Operator
            The Operator to be approximated.
        maxiter: int or None (defaults 300)
            Number of iteration (equals number of eigenvalues).
            If set to None, stops at A.shape[0]

        Returns
        -------
        A LanczosAlgorithm instance. To get the approximated Operator,
        calling this instance is required.

        Notes
        -----
        Starting point is a normalized random vector so results may
        differ from one call to another with the same input parameters.

        The Operator approximation is returned as a
        EigendecompositionOperator which can be easily inverted.
        """
        self.A = asoperator(A)
        self.n = self.A.shape[0]
        self.maxiter = maxiter
        self.norm = lambda x: np.sqrt(np.dot(x, x)) #XXX //ise me
        stop_condition = MaxIterationStopCondition(maxiter)

        IterativeAlgorithm.__init__(self, normal_stop_condition=stop_condition)
        self.v0 = v0
        # tridiagonal matrix coefficients
        self.alpha = np.zeros(self.maxiter)
        self.beta = np.zeros(self.maxiter)
        self.vectors = np.zeros((self.maxiter+1, self.n))

    def initialize(self):
        IterativeAlgorithm.initialize(self)
        if self.v0 is None:
            v0 = np.random.randn(self.n)
        else:
            v0 = self.v0.ravel()
        v0 /= self.norm(v0)
        self.vectors[0] = v0
        self.alpha[...] = 0
        self.beta[...] = 0

    def iteration(self):
        n = self.niterations
        v = self.vectors[n-1]
        v_new = self.vectors[n]
        self.A.matvec(v, out=v_new)

        # orthogonalisation
        if n > 1:
            v_new -= self.beta[n-2] * self.vectors[n-2]

        alpha = np.dot(v_new, v)
        v_new -= alpha * v
        beta = self.norm(v_new)
        v_new /= beta

        # update
        self.alpha[n-1] = alpha
        self.beta[n-1] = beta

    def finalize(self):
        """
        Convert alpha and beta to a TridiagonalOperator and perform
        eigendecomposition.

        """
        T = TridiagonalOperator(self.alpha, self.beta[:-1])
        # use band matrix eigendecomposition as LAPACK's SEV routines (?STEV*)
        # for symmetric tridiagonal matrices are not available in scipy 0.10
        E = T.toband().eigen()

        # multiply T eigenvectors with lanczos vectors
        w = E.eigenvalues
        v = np.dot(self.vectors[:-1, :].T, E.eigenvectors)

        return EigendecompositionOperator(v=v, w=w)
