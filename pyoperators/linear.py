from __future__ import division

import numpy as np

from scipy.sparse.linalg import eigsh

from .decorators import linear, real, symmetric, inplace
from .core import Operator, BlockRowOperator, CompositionOperator, DiagonalOperator, ReductionOperator, asoperator
from .utils import isscalar

__all__ = [
    'BandOperator',
    'EigendecompositionOperator',
    'IntegrationTrapezeWeightOperator',
    'PackOperator',
    'SumOperator',
    'SymmetricBandOperator',
    'TridiagonalOperator',
    'UnpackOperator',
]


class IntegrationTrapezeWeightOperator(BlockRowOperator):
    """
    Return weights as a block row operator to perform trapeze integration.

    This operator can be used to integrate over X the bivariate function 
        f = f(X,Y).
    Let's assume f is sampled at n abscissa x_n non necessarily equally spaced
        f_i(Y) = f(x_i, Y).
    The operator IntegrationTrapezeWeightOperator returns a block row operator
        W = [ w_1 * I ... w_n * I]
    such that, given the block column operator
            [ f_1 ]
        F = [ ... ]
            [ f_n ],
    the product
        W * F = w_1 * f_1 + ... + w_n * f_n
    performs a trapeze integration of f(X,Y) over the bins [x_i,x_(i+1)]
    for i in 1..n-1.

    Example
    -------
    >>> f = np.power
    >>> x = [0.5,1,2,4]
    >>> F = BlockColumnOperator([Operator(lambda i,o,v=v:f(v,i,o),
    ...                         flags='square') for v in x], new_axisout=0)
    >>> W = IntegrationTrapezeWeightOperator(x)
    >>> int_f = W * F
    >>> int_f([0,1,2])
    array([  3.5   ,   7.875 ,  22.8125])
    >>> [ trapz(f(x,a), x) for a in [0,1,2] ]
    [3.5, 7.875, 22.8125]

    """
    def __init__(self, x, new_axisin=0, **keywords):
        x = np.asarray(x)
        if x.size < 2:
            raise ValueError('At least two abscissa are required.')
        if np.any(np.diff(x) < 0) and np.any(np.diff(x) > 0):
            raise ValueError('The abscissa are not monotonous.')

        w = np.empty_like(x)
        w[0] = 0.5 * (x[1] - x[0])
        w[1:-1] = 0.5 * (x[2:]-x[:-2])
        w[-1] = 0.5 * (x[-1] - x[-2])
        BlockRowOperator.__init__(self, list(w), new_axisin=new_axisin,
                                  **keywords)


@linear
@real
@inplace
class PackOperator(Operator):
    """
    Convert an ndarray into a vector, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=self.mask.shape,
                          shapeout=np.sum(self.mask), **keywords)
        #XXX .T does not share the same mask...
        self.set_rule('.T', lambda s: UnpackOperator(~s.mask, dtype=s.dtype))
        self.set_rule('.T.', '1', CompositionOperator)

    def direct(self, input, output):
        output[...] = input[self.mask]


@linear
@real
class UnpackOperator(Operator):
    """
    Convert a vector into an ndarray, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=np.sum(self.mask),
                          shapeout=self.mask.shape, **keywords)
        self.set_rule('.T', lambda s: PackOperator(~s.mask, dtype=s.dtype))
        self.set_rule('.T.', '1', CompositionOperator)

    def direct(self, input, output):
        output[...] = 0
        output[self.mask] = input


@linear
class SumOperator(ReductionOperator):
    """
    Sum-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = SumOperator()
    >>> op([1,2,3])
    array(6)

    """
    def __init__(self, axis=None, dtype=None, skipna=True, **keywords):
        if np.__version__ < '1.7':
            func = np.nansum if skipna else np.add
        else:
            func = np.add
        ReductionOperator.__init__(self, func, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)

    def transpose(self, input, output):
        if self.axis is None:
            shape = None
        elif self.axis == -1:
            shape = input.shape + (1,)
        else:
            shape = input.shape[:self.axis] + (1,) + input.shape[self.axis:]
        output[...] = input.reshape(shape)


@linear
class TridiagonalOperator(Operator):
    def __init__(self, diag, subdiag, superdiag=None, **kwargs):
        """
        Store a tridiagonal operator in the form of 3 arrays

        Parameters
        ----------
        shape : length 2 tuple.
            The shape of the operator.

        diag : ndarray of size shape[0]
            The diagonal of the matrix.

        subdiag : ndarray of size shape[0] - 1
            The subdiagonal of the matrix.

        superdiag : ndarray of size shape[0] - 1 or None (default:None)
            The superdiagonal of the matrix.
            If set to None, the Operator is symmetric.

        Returns
        -------
        A Tridiagonal matrix operator instance.

        Exemple
        -------
        >>> import operators
        >>> T = operators.TridiagonalOperator([1, 2, 3], [4, 5], [6, 7])
        >>> T.todense()
        array([[1, 6, 0],
               [4, 2, 7],
               [0, 5, 3]])
        """
        self.diag = np.asarray(diag)
        shapein = (len(diag),)
        # define subdiag
        if np.iterable(subdiag) and len(subdiag) + 1 != len(diag):
            if isscalar(subdiag):
                subdiag *= np.ones(len(diag) - 1)
            else:
                raise ValueError("subdiagonal should be the length of the diagonal minus one or a scalar.")
        self.subdiag = np.asarray(subdiag)
        # define superdiag
        if superdiag is None:
            self.superdiag = self.subdiag
        else:
            self.superdiag = np.asarray(superdiag)
        if np.iterable(self.superdiag) and len(self.superdiag) + 1 != len(diag):
            if isscalar(self.superdiag):
                self.superdiag *= np.ones(len(diag) - 1)
            else:
                raise ValueError("superdiagonal should be the length of the diagonal minus one.")
        self.dtype = np.find_common_type([],[self.diag.dtype,
                                             self.subdiag.dtype,
                                             self.superdiag.dtype])
        self.kwargs = kwargs
        flags = {"symmetric":np.all(self.subdiag == self.superdiag)}
        Operator.__init__(self, shapein=shapein, dtype=self.dtype, flags=flags,
                          **kwargs)
        self.set_rule('.T', lambda s: TridiagonalOperator(self.diag,
                      self.superdiag, self.subdiag))
        self.set_rule('.C', lambda s: TridiagonalOperator(np.conj(self.diag),
                      np.conj(self.subdiag), np.conj(self.superdiag)))

    def direct(self, input, output):
        output[:] = self.diag * input
        output[:-1] += self.superdiag * input[1:]
        output[1:] += self.subdiag * input[:-1]

    def transpose(self, input, output):
        output = self.diag * input
        output[:-1] += self.subdiag * input[1:]
        output[1:] += self.superdiag * input[:-1]

    def __repr__(self):
        r = [repr(self.diag), repr(self.subdiag)]
        if self.subdiag is not self.superdiag:
            r += ['superdiag=' + repr(self.superdiag)]
        if any([len(_) > 70 for _ in r]):
            sep = ',\n'
            r[0] = '\n' + r[0]
        else:
            sep = ', '
        return self.__name__ + '(' + sep.join(r) + ')'

    def todense(self):
        out = np.zeros(self.shape, dtype=self.dtype)
        out += np.diag(self.diag)
        out += np.diag(self.subdiag, -1)
        out += np.diag(self.superdiag, 1)
        return out

    def __getitem__(self, y):
        # if tuple work on two dimensions
        if isinstance(y, tuple):
            # test dimension
            if len(y) > 2:
                raise IndexError("This is a 2-dimensional array.")
            yi, yj = y
            # single element case
            if isinstance(yi, int) and isinstance(yj, int):
                n = self.shape[0]
                i, j = yi % n , yj % n
                # outside
                if np.abs(i - j) > 1:
                    return self.dtype.type(0)
                # subdiag
                elif i == j + 1:
                    # border case
                    if i == self.shape[0] - 1:
                        return self.dtype.type(0)
                    else:
                        return self.subdiag[i]
                # superdiag
                elif i == j - 1:
                    # border case
                    if i == self.shape[0]:
                        return self.dtype.type(0)
                    else:
                        return self.superdiag[i]
                # diag
                else:
                    return self.diag[i]
            # case of tuple of length 1
            elif len(y) == 1:
                return self.__getitem__(self, y[0])
            # get a column
            elif yi == slice(None, None) and isinstance(yj, int):
                x = np.zeros(self.shape[1], dtype=self.dtype)
                x[yj] = 1.
                return self * x
            # general case: no better way than todense
            else:
                d = self.todense()
                return d[y]
        # Work on lines : same cost as recasting to a dense matrix as
        # all columns need to be accessed.
        else:
            d = self.todense()
            return d[y]

    def toband(self):
        """
        Convert the TridiagonalOperator into a BandOperator
        """
        if not self.flags.symmetric:
            kl, ku = 1, 1
            n = self.shape[1]
            ab = np.zeros((kl + ku + 1, n))
            diags = (self.subdiag, self.diag, self.superdiag)
            for i, d in zip((-1, 0, 1), diags):
                ab[_band_diag(ku, i)] = d
            return BandOperator(ab, kl, ku, **self.kwargs)
        else:
            u = 2 # tridiagonal
            n = self.shape[0]
            # convert to ab format (lower)
            ab = np.zeros((u, n))
            ab[0] = self.diag
            ab[1, :-1] = self.subdiag
            return SymmetricBandOperator(ab, lower=True, dtype=self.dtype)

@linear
class BandOperator(Operator):
    """
    Store a band matrix in ab format as defined in LAPACK
    documentation.

    a[i, j] is stored in ab[ku + 1 + i - j, j]

    for max(1, j -ku) < i < min(m, j + kl)

    Band storage of A (5, 5), kl = 2, ku = 1 :

     *  a01 a12 a23 a34
    a00 a11 a22 a33 a44
    a10 a21 a32 a43  *
    a20 a31 a42  *   *

    Arguments
    ----------
    shape : 2-tuple
        Shape of the dense matrix equivalent.
    kl : int
        Number of subdiagonals
    ku : int
        Number of superdiagonals

    Notes
    -----
    For a description of band matrices see LAPACK doc :

    http://www.netlib.org/lapack/lug/node124.html

    """
    def __init__(self, ab, kl, ku, **kwargs):
        """
        Generate a BandOperator instance

        Arguments
        ---------
        shape : 2-tuple
           The shape of the operator
        ab : ndarray with ndim == 2
           Store the bands of the matrix using LAPACK storage scheme.
        kl : int
            Number of subdiagonals
        ku : int
            Number of superdiagonals
        """
        shapein = ab.shape[1]
        self.ab = ab
        self.kl = kl
        self.ku = ku
        self.kwargs = kwargs

        return Operator.__init__(self, shapein=shapein, **kwargs)

    def direct(self, x, out):
        # diag
        out[:] = self.ab[self.ku] * x
        # upper part
        for i in xrange(self.ku):
            j = self.ku - i
            out[:-j] += self.ab[i, j:] * x[j:]
        for i in xrange(self.ku, self.kl + self.ku):
            # lower part
            out[i:] += self.ab[i + 1, :-i] * x[:-i]

    def transpose(self, x, out):
        rab = self._rab
        rkl, rku = self.ku, self.kl
        # diag
        out = self.rab[self.ku] * x
        # upper part
        for i in xrange(rku):
            j = rku - i
            out[:-j] += rab[i, j:] * x[j:]
        for i in xrange(rku, rkl + rku):
            # lower part
            out[i:] += rab[i + 1, :-i] * x[:-i]

    def diag(self, i=0):
        """
        Returns the i-th diagonal (subdiagonal if i < 0, superdiagonal
        if i >0).
        """
        return self.ab[_band_diag(self.ku, i)]

    @property
    def rab(self):
        """
        Output the ab form of the transpose operator.
        """
        ab = self.ab
        kl, ku = self.kl, self.ku
        rku, rkl = kl, ku
        rab = np.zeros(ab.shape, dtype=ab.dtype)
        for i in xrange(- kl, ku + 1):
            rab[_band_diag(rku, -i)] = self.diag(i)
        return rab

def _band_diag(ku, i=0):
    """
    Return a slice to get the i-th line of a band operator
    """
    # diagonal
    if i == 0:
        return slice(ku, ku + 1)
    # superdiagonal
    if i > 0:
        return (slice(ku - i, ku - i + 1, None), slice(i, None, None))
    # subdiagonal
    if i < 0:
        return (slice(ku - i, ku - i + 1, None), slice(None, i, None))

class LowerTriangularOperator(BandOperator):
    """
    A BandOperator with no upper diagonals (ku=0)
    """
    def __init__(self, ab, **kwargs):
        kl = ab.shape[0] - 1
        ku = 0
        BandOperator.__init__(self, ab, kl, ku, **kwargs)

class UpperTriangularOperator(BandOperator):
    """
    A BandOperator with no lower diagonals (kl=0)
    """
    def __init__(self, ab, **kwargs):
        kl = 0
        ku = ab.shape[0] - 1
        BandOperator.__init__(self, ab, kl, ku, **kwargs)

@symmetric
class SymmetricBandOperator(Operator):
    """
    SymmetricBandOperator do not store diagonal datas in the same
    format as BandOperator does. This is not a subclass of
    BandOperator.
    """
    def __init__(self, ab, lower=True, **kwargs):
        shapein = ab.shape[1]
        self.ab = ab
        self.lower = lower
        self.kwargs = kwargs

        return Operator.__init__(self, shapein=shapein, **kwargs)

    def direct(self, x, out):
        out[:] = self.ab[0] * x
        for i in xrange(1, self.ab.shape[0]):
            # upper part
            out[:-i] += self.ab[i, :-i] * x[i:]
            # lower part
            out[i:] += self.ab[i, :-i] * x[:-i]

    @property
    def rab(self):
        return self.ab

    def eigen(self, eigvals_only=False, overwrite_a_band=False, select='a',
              select_range=None, max_ev=0):
        """
        Solve real symmetric or complex hermitian band matrix
        eigenvalue problem.

        Uses scipy.linalg.eig_banded function.
        """
        from scipy.linalg import eig_banded

        w, v = eig_banded(self.ab, lower=self.lower,
                          eigvals_only=eigvals_only,
                          overwrite_a_band=overwrite_a_band,
                          select=select,
                          select_range=select_range,
                          max_ev=max_ev)
        return EigendecompositionOperator(w=w, v=v, **self.kwargs)

    def cholesky(self, overwrite_ab=False):
        """
        Chlesky decomposition.
        Operator needs to be positive-definite.

        Uses scipy.linalg.cholesky_banded.

        Returns a matrix in ab form
        """
        from scipy.linalg import cholesky_banded

        ab_chol = cholesky_banded(self.ab,
                               overwrite_ab=overwrite_ab,
                               lower=self.lower)
        if self.lower:
            out = LowerTriangularOperator(self.shape, ab_chol, **self.kwargs)
        else:
            out = UpperTriangularOperator(self.shape, ab_chol, **self.kwargs)
        return out

@symmetric
class EigendecompositionOperator(Operator):
    """
    Define a symmetric Operator from the eigendecomposition of another
    symmetric Operator. This can be used as an approximation for the
    operator.

    Inputs
    -------

    A: LinearOperator (default: None)
      The LinearOperator to approximate.
    v: 2d ndarray (default: None)
      The eigenvectors as given by arpack.eigsh
    w: 1d ndarray (default: None)
      The eigenvalues as given by arpack.eigsh
    **kwargs: keyword arguments
      Passed to the arpack.eigsh function.

    You need to specify either A or v and w.

    Returns
    -------

    An EigendecompositionOperator instance, which is a subclass of
    Operator.

    Notes
    -----

    This is really a wrapper for
    scipy.sparse.linalg.eigen.arpack.eigsh
    """
    def __init__(self, A=None, v=None, w=None, **kwargs):
        if v is None or w is None:
            w, v = eigsh(A, return_eigenvectors=True, **kwargs)
            kwargs['dtype'] = A.dtype
        else:
            kwargs['dtype'] = v.dtype
        self.W = DiagonalOperator(w)
        self.V = asoperator(v)
        self.M = self.V * self.W * self.V.T
        # store some information
        self.eigenvalues = w
        self.eigenvectors = v
        self.kwargs = kwargs
        Operator.__init__(self, shapein=self.M.shapein, direct=self.M.direct,
                          **kwargs)
        self.set_rule('.I', lambda s: s ** -1)

    def det(self):
        """
        Output an approximation of the determinant from the
        eigenvalues.
        """
        return np.prod(self.eigenvalues)

    def logdet(self):
        """
        Output the log of the determinant. Useful as the determinant
        of large matrices can exceed floating point capabilities.
        """
        return np.sum(np.log(self.eigenvalues))

    def __pow__(self, n):
        """
        Raising an eigendecomposition to an integer power requires
        only raising the eigenvalues to this power.
        """
        return EigendecompositionOperator(v=self.eigenvectors,
                                          w=self.eigenvalues ** n,
                                          **self.kwargs)

    def inv(self):
        """
        Returns the pseudo-inverse of the operator.
        """
        return self ** -1

    def trace(self):
        return np.sum(self.eigenvalues)

    def cond(self):
        """
        Output an approximation of the condition number by taking the
        ratio of the maximum over the minimum eigenvalues, removing
        the zeros.

        For better approximation of the condition number, one should
        consider generating the eigendecomposition with the keyword
        which='BE', in order to have a correct estimate of the small
        eigenvalues.
        """
        nze = self.eigenvalues[self.eigenvalues != 0]
        return nze.max() / nze.min()
