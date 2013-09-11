from __future__ import division

import multiprocessing
import numexpr
import numpy as np

try:
    import pyfftw
except:
    pass
from itertools import izip
from scipy.sparse.linalg import eigsh

from .decorators import inplace, linear, real, square, symmetric
from .core import (
    Operator,
    BlockRowOperator,
    BroadcastingOperator,
    CompositionOperator,
    DenseOperator,
    DiagonalOperator,
    ReductionOperator,
    Variable,
    X,
    _pool,
)
from .memory import empty
from .utils import cast, complex_dtype, float_dtype, ifirst, izip_broadcast, tointtuple

__all__ = [
    'BandOperator',
    'DiagonalNumexprOperator',
    'DiagonalNumexprNonSeparableOperator',
    'DifferenceOperator',
    'EigendecompositionOperator',
    'IntegrationTrapezeWeightOperator',
    'PackOperator',
    'Rotation2dOperator',
    'SumOperator',
    'SymmetricBandOperator',
    'SymmetricBandToeplitzOperator',
    'TridiagonalOperator',
    'UnpackOperator',
]


class DiagonalNumexprOperator(DiagonalOperator):
    """
    DiagonalOperator whose diagonal elements are calculated on the fly using
    the numexpr package and that can be separated when added or multiplied
    to a block operator.

    Note
    ----
    When such instance is added or multiplied to another DiagonalOperator
    (or subclass, such as an instance of this class), an algebraic
    simplification takes place, which results in a regular (dense) diagonal
    operator.

    Example
    -------
    >>> alpha = np.arange(100.)
    >>> d = DiagonalNumexprOperator(alpha, '(x/x0)**data', {'x':1.2, 'x0':1.})

    """

    def __init__(
        self, data, expr, global_dict=None, var='data', dtype=float, **keywords
    ):
        if not isinstance(expr, str):
            raise TypeError('The second argument is not a string expression.')
        if numexpr.__version__ < '2.0.2':
            keywords['flags'] = self.validate_flags(
                keywords.get('flags', {}), inplace=False
            )
        BroadcastingOperator.__init__(self, data, dtype=dtype, **keywords)
        self.expr = expr
        self.var = var
        self.global_dict = global_dict
        self._global_dict = {} if global_dict is None else global_dict.copy()
        self._global_dict[var] = (
            self.data.T if self.broadcast == 'rightward' else self.data
        )

    def direct(self, input, output):
        if self.broadcast == 'rightward':
            input = input.T
            output = output.T
        numexpr.evaluate(
            '(' + self.expr + ') * input', global_dict=self._global_dict, out=output
        )

    def get_data(self):
        local_dict = {self.var: self.data}
        return numexpr.evaluate(
            self.expr, local_dict=local_dict, global_dict=self.global_dict
        )

    @staticmethod
    def _rule_block(self, op, shape, partition, axis, new_axis, func_operation):
        if type(self) is not DiagonalNumexprOperator:
            return None
        return DiagonalOperator._rule_block(
            self,
            op,
            shape,
            partition,
            axis,
            new_axis,
            func_operation,
            self.expr,
            global_dict=self.global_dict,
            var=self.var,
        )


class DiagonalNumexprNonSeparableOperator(DiagonalOperator):
    """
    DiagonalOperator whose diagonal elements are calculated on the fly using
    the numexpr package.

    Notes
    -----
    - When such an instance is added or multiplied to another DiagonalOperator
    (or subclass, such as an instance of this class), an algebraic
    simplification takes place, which results in a regular (dense) diagonal
    operator.
    - This operator can not be separated so that each part handles a block
    of a block operator. Also, rightward broadcasting cannot be used. If one of
    these properties is desired, use the class DiagonalNumexprOperator.
    - If the operator's input shape is not specified, its inference costs
    an evaluation of the expression.

    Example
    -------
    >>> alpha = np.arange(100.)
    >>> d = DiagonalNumexprNonSeparableOperator(
    ...     '(x/x0)**alpha', {'alpha': alpha, 'x': 1.2,'x0': 1})

    """

    def __init__(self, expr, global_dict=None, dtype=float, **keywords):
        if not isinstance(expr, str):
            raise TypeError('The first argument is not a string expression.')
        if 'broadcast' in keywords and keywords['broadcast'] == 'rightward':
            raise ValueError(
                'The class DiagonalNumexprNonSeparableOperator does not handle'
                ' rightward broadcasting. Use the class DiagonalNumexprOperato'
                'r for this purpose.'
            )
        if 'broadcast' not in keywords or keywords['broadcast'] != 'leftward':
            keywords['broadcast'] = 'disabled'
        self.expr = expr
        self.global_dict = global_dict
        if (
            'shapein' not in keywords
            and 'shapeout' not in keywords
            and keywords['broadcast'] == 'disabled'
        ):
            keywords['shapein'] = self.get_data().shape
        if numexpr.__version__ < '2.0.2':
            keywords['flags'] = self.validate_flags(
                keywords.get('flags', {}), inplace=False
            )
        BroadcastingOperator.__init__(self, 0, dtype=dtype, **keywords)

    def direct(self, input, output):
        numexpr.evaluate(
            '(' + self.expr + ') * input', global_dict=self.global_dict, out=output
        )

    def get_data(self):
        return numexpr.evaluate(self.expr, global_dict=self.global_dict)

    @staticmethod
    def _rule_left_block(self, op, cls):
        return None

    @staticmethod
    def _rule_right_block(self, op, cls):
        return None


@linear
class EinsteinSumOperator(Operator):
    """
    Einstein sum operator.

    Example
    -------
    >>> op = EinsteinSumOperator('ii->i', X, shapein=(4,4))
    >>> op(np.arange(16).reshape(4,4))
    [0., 5., 10., 15.]

    """

    def __init__(self, subscripts, *operands, **keywords):
        subscripts = subscripts.replace(' ', '')
        if '->' not in subscripts:
            raise NotImplementedError(
                'Unlike einsum, the output subscript cannot yet be specified.'
            )
        in_, out = subscripts.split('->')
        ins = in_.split(',')
        if len(operands) == len(ins) - 1 and X not in operands:
            operands = operands + (X,)
        if len(ins) != len(operands):
            raise ValueError(
                "The number of input operands '{0}' does not match that from t"
                "he subscripts '{1}'.".format(len(operands), len(ins))
            )
        self.iX = ifirst(operands, lambda x: x is X)
        self.operands = operands
        self.subscripts = subscripts
        self.lsubscripts = tuple([self._get_subscript_as_list(s) for s in ins + [out]])
        Operator.__init__(self, **keywords)
        tsubscripts = self._get_reversed_subscripts()
        if self.subscripts == '...jk,...k->...j':
            tsubscripts = '...kj,...k->...j'
        elif self.subscripts == '...kj,...k->...j':
            tsubscripts = '...jk,...k->...j'
        else:
            raise NotImplementedError()
        self.set_rule(
            '.T',
            lambda s: ReverseOperatorFactory(
                EinsteinSumOperator, s, tsubscripts, *self.operands
            ),
        )

    def direct(self, input, output):
        operands = list(self.operands)
        operands[self.iX] = input
        np.einsum(self.subscripts, *operands, out=output)

    def reshapein(self, shapein):
        shapes = [
            None if isinstance(o, Variable) else o.shape for o in self.operands
        ] + [None]
        shapes[self.iX] = shapein
        return self._get_shape(shapes, -1)

    def reshapeout(self, shapeout):
        shapes = [
            None if isinstance(o, Variable) else o.shape for o in self.operands
        ] + [shapeout]
        return self._get_shape(shapes, self.iX)

    @staticmethod
    def _enumerate_ellipsis(l):
        """
        Enumerate a sequence that may include an Ellipsis, in which case
        a slice is returned and negative indices afterwards.

        Example
        -------
        >>> list(enumerate_ellipsis(['a', 'b', Ellipsis, 'c']))
        [0, 1, slice(2, -1, None), -1]

        """
        i = 0
        for x in l:
            if x is not Ellipsis:
                yield i, l[i]
                i += 1
            else:
                yield slice(i, i - len(l) + 1), Ellipsis
                i -= len(l) - 1

    def _get_reversed_subscripts(self):
        ls = self.lsubscripts
        ls_reversed = [l[::-1] for l in ls[-2::-1]] + [ls[-1][::-1]]
        ls_reversed = [
            ''.join(['...' if a is Ellipsis else a for a in l]) for l in ls_reversed
        ]
        return ','.join(ls_reversed[:-1]) + '->' + ls_reversed[-1]

    def _get_shape(self, shapes, i):
        """Infer ith shape from the shapes of the other operands."""
        shape = []
        # print 'shapes', shapes
        for iaxis, subscript in self._enumerate_ellipsis(self.lsubscripts[i]):
            # find this subscript elsewhere
            # print 'iaxis, subscript',  iaxis, subscript
            try:
                for iother, subscripts_other in enumerate(self.lsubscripts):
                    # print 'iother, sub_other', iother, subscripts_other, 'skipped' if iother == i or shapes[iother] is None else 'looking...'
                    if iother == i or shapes[iother] is None:
                        continue
                    for iaxis_other, subscript_other in self._enumerate_ellipsis(
                        subscripts_other
                    ):
                        # print 'iaxis_other, subscript_other:', iaxis_other, subscript_other
                        if subscript == subscript_other:
                            new_dims = tointtuple(shapes[iother][iaxis_other])
                            shape.extend(new_dims)
                            raise StopIteration()
                if i == -1:
                    raise ValueError(
                        "Einstein sum subscripts include an output subscript '"
                        "{0}' which never appeared in an input.".format(subscript)
                    )
                return None
            except StopIteration:
                pass
        return tuple(shape)

    def _get_subscript_as_list(self, s):
        s = s.replace('...', '.')
        return [l if l != '.' else Ellipsis for l in s]


@real
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
    >>> from pyoperators import BlockColumnOperator
    >>> f = np.power
    >>> x = [0.5,1,2,4]
    >>> F = BlockColumnOperator(
    ...         [Operator(lambda i, o, v=v: f(v, i, o), flags='square')
    ...          for v in x], new_axisout=0)
    >>> W = IntegrationTrapezeWeightOperator(x)
    >>> int_f = W * F
    >>> int_f([0,1,2])
    array([  3.5   ,   7.875 ,  22.8125])
    >>> [np.trapz(f(x, a), x) for a in [0, 1, 2]]
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
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[-1] = 0.5 * (x[-1] - x[-2])
        BlockRowOperator.__init__(self, list(w), new_axisin=new_axisin, **keywords)


@real
@linear
@inplace
class PackOperator(Operator):
    """
    Convert an ndarray into a vector, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(
            self, shapein=self.mask.shape, shapeout=np.sum(self.mask), **keywords
        )
        # XXX .T does not share the same mask...
        self.set_rule('.T', lambda s: UnpackOperator(~s.mask, dtype=s.dtype))
        self.set_rule('.T.', '1', CompositionOperator)

    def direct(self, input, output):
        output[...] = input[self.mask]


@real
@linear
class UnpackOperator(Operator):
    """
    Convert a vector into an ndarray, under the control of a mask.
    """

    def __init__(self, mask, **keywords):
        self.mask = ~np.array(mask, np.bool8, copy=False)
        Operator.__init__(
            self, shapein=np.sum(self.mask), shapeout=self.mask.shape, **keywords
        )
        self.set_rule('.T', lambda s: PackOperator(~s.mask, dtype=s.dtype))
        self.set_rule('.T.', '1', CompositionOperator)

    def direct(self, input, output):
        output[...] = 0
        output[self.mask] = input


@real
@linear
class Rotation2dOperator(DenseOperator):
    """
    2-d rotation operator.

    Parameters
    ----------
    angle : float, array-like
        The counter-clockwise rotation angle, in radians.
    degrees : bool, optional
        If set, the angle input is in degrees, instead of radians.

    Example
    -------
    >>> r = Rotation2dOperator([45, 90], degrees=True)
    >>> r([1, 0])
    >>> print r([1, 0])
    [[  7.07106781e-01   7.07106781e-01]
     [  6.12323400e-17   1.00000000e+00]]

    """

    def __init__(self, angle, degrees=False, dtype=None, **keywords):
        angle = np.asarray(angle)
        if dtype is None:
            dtype = float_dtype(angle.dtype)
        if degrees:
            angle = np.radians(angle)
        cosa = np.cos(angle)
        sina = np.sin(angle)
        m = np.array([[cosa, -sina], [sina, cosa]], dtype=dtype)
        for i in range(angle.ndim):
            m = np.rollaxis(m, -1)
        keywords['flags'] = self.validate_flags(
            keywords.get('flags', {}), orthogonal=m.ndim == 2
        )
        DenseOperator.__init__(self, m, **keywords)


@real
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
        if np.__version__ < '1.8':
            func = np.nansum if skipna else np.add
        else:
            func = np.add
        ReductionOperator.__init__(
            self, func, axis=axis, dtype=dtype, skipna=skipna, **keywords
        )

    def transpose(self, input, output):
        if self.axis is None:
            shape = None
        elif self.axis == -1:
            shape = input.shape + (1,)
        else:
            shape = input.shape[: self.axis] + (1,) + input.shape[self.axis :]
        output[...] = input.reshape(shape)


@linear
@square
class TridiagonalOperator(Operator):
    def __init__(
        self, diagonal, subdiagonal, superdiagonal=None, dtype=None, **keywords
    ):
        """
        Store a tridiagonal operator in the form of 3 arrays.

        TODO: there is no such gtmv in BLAS. Implement fast (r)matvec or
              investigate making it a BandOperator subclass
        =====

        Parameters
        ----------
        diagonal : ndarray of size N
            The diagonal of the matrix.

        subdiagonal : ndarray of size the size N-1
            The subdiagonal of the matrix.

        superdiagonal : ndarray of size the size N-1
            The superdiagonal of the matrix. If it is None, the superdiagonal
            is assumed to be the conjugate of the subdiagonal.

        Exemple
        -------
        >>> import operators
        >>> T = operators.TridiagonalOperator([1, 2, 3], [4, 5], [6, 7])
        >>> T.todense()
        array([[1, 6, 0],
               [4, 2, 7],
               [0, 5, 3]])

        """
        diagonal, subdiagonal, superdiagonal = cast(
            [diagonal, subdiagonal, superdiagonal], dtype=dtype
        )
        dtype = diagonal.dtype

        if diagonal.ndim != 1:
            raise ValueError('The diagonal must be a 1-dimensional array.')
        if subdiagonal.ndim != 1:
            raise ValueError('The diagonal must be a 1-dimensional array.')
        if superdiagonal is not None and superdiagonal.ndim != 1:
            raise ValueError('The diagonal must be a 1-dimensional array.')

        shapein = diagonal.size
        if subdiagonal.size not in (1, shapein - 1):
            raise ValueError(
                'The sub diagonal should be the length of the diagonal minus o'
                'ne or a scalar.'
            )
        if superdiagonal is not None and superdiagonal.size not in (1, shapein - 1):
            raise ValueError(
                'The super diagonal should be the length of the d'
                'iagonal minus one or a scalar.'
            )

        if superdiagonal is None:
            superdiagonal = subdiagonal.conj()

        self.diagonal = diagonal
        self.subdiagonal = subdiagonal
        self.superdiagonal = superdiagonal

        flags = {
            'real': dtype.kind != 'c',
            'symmetric': np.allclose(self.subdiagonal, self.superdiagonal),
            'hermitian': np.allclose(self.diagonal.imag, 0)
            and np.allclose(self.subdiagonal, self.superdiagonal.conj()),
        }
        keywords['flags'] = flags
        keywords['shapein'] = shapein

        Operator.__init__(self, dtype=dtype, **keywords)
        self.set_rule(
            '.T',
            lambda s: TridiagonalOperator(s.diagonal, s.superdiagonal, s.subdiagonal),
        )
        self.set_rule(
            '.C',
            lambda s: TridiagonalOperator(
                s.diagonal.conj(), s.subdiagonal.conj(), s.superdiagonal.conj()
            ),
        )
        self.set_rule(
            '.H',
            lambda s: TridiagonalOperator(
                s.diagonal.conj(), s.superdiagonal.conj(), s.subdiagonal.conj()
            ),
        )

    def direct(self, input, output):
        output[:] = self.diagonal * input
        output[:-1] += self.superdiagonal * input[1:]
        output[1:] += self.subdiagonal * input[:-1]

    def transpose(self, input, output):
        output = self.diagonal * input
        output[:-1] += self.subdiagonal * input[1:]
        output[1:] += self.superdiagonal * input[:-1]

    def todense(self):
        # XXX optimize me
        out = np.zeros(self.shape, dtype=self.dtype)
        out += np.diag(self.diagonal)
        out += np.diag(self.subdiagonal, -1)
        out += np.diag(self.superdiagonal, 1)
        return out

    def toband(self):
        """
        Convert the TridiagonalOperator into a BandOperator
        """
        if not self.flags.symmetric:
            kl, ku = 1, 1
            n = self.shape[1]
            ab = np.zeros((kl + ku + 1, n), self.dtype)
            diags = (self.subdiagonal, self.diagonal, self.superdiagonal)
            for i, d in zip((-1, 0, 1), diags):
                ab[_band_diag(ku, i)] = d
            return BandOperator(ab, kl, ku)
        else:
            u = 2  # tridiagonal
            n = self.shape[0]
            # convert to ab format (lower)
            ab = np.zeros((u, n), self.dtype)
            ab[0] = self.diagonal
            ab[1, :-1] = self.subdiagonal
            return SymmetricBandOperator(ab, lower=True)


@linear
@square
class BandOperator(Operator):
    """
    Store a band matrix in ab format as defined in LAPACK
    documentation.

    TODO:direct and transpose methods should call BLAS2 gbmv (not yet in scipy)
    =====

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
        for i in xrange(-kl, ku + 1):
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

    TODO: direct method should call BLAS2 sbmv (not yet in scipy)
    =====

    """

    def __init__(self, ab, lower=True, **kwargs):
        kwargs['shapein'] = ab.shape[1]
        self.ab = ab
        self.lower = lower
        self.kwargs = kwargs
        return Operator.__init__(self, **kwargs)

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

    def eigen(
        self,
        eigvals_only=False,
        overwrite_a_band=False,
        select='a',
        select_range=None,
        max_ev=0,
    ):
        """
        Solve real symmetric or complex hermitian band matrix
        eigenvalue problem.

        Uses scipy.linalg.eig_banded function.
        """
        from scipy.linalg import eig_banded

        w, v = eig_banded(
            self.ab,
            lower=self.lower,
            eigvals_only=eigvals_only,
            overwrite_a_band=overwrite_a_band,
            select=select,
            select_range=select_range,
            max_ev=max_ev,
        )
        return EigendecompositionOperator(w=w, v=v, **self.kwargs)

    def cholesky(self, overwrite_ab=False):
        """
        Chlesky decomposition.
        Operator needs to be positive-definite.

        Uses scipy.linalg.cholesky_banded.

        Returns a matrix in ab form
        """
        from scipy.linalg import cholesky_banded

        ab_chol = cholesky_banded(self.ab, overwrite_ab=overwrite_ab, lower=self.lower)
        if self.lower:
            out = LowerTriangularOperator(self.shape, ab_chol, **self.kwargs)
        else:
            out = UpperTriangularOperator(self.shape, ab_chol, **self.kwargs)
        return out


@real
@linear
@symmetric
@inplace
class SymmetricBandToeplitzOperator(Operator):
    """
    The SymmetricBandToeplitz operator for symmetric band Toeplitz matrices.

    The vector product is implemented using the FFTW library, so it scales
    as O(nlogn) operations.

    Example
    -------
    >>> N = SymmetricBandToeplitzOperator(5, [3, 2, 1, 1])
    >>> print N.todense().astype(int)
    [[3 2 1 1 0]
     [2 3 2 1 1]
     [1 2 3 2 1]
     [1 1 2 3 2]
     [0 1 1 2 3]]

    """

    def __init__(
        self,
        shapein,
        firstrow,
        dtype=None,
        fftw_flag='FFTW_MEASURE',
        nthreads=None,
        **keywords,
    ):
        shapein = tointtuple(shapein)
        if dtype is None:
            dtype = float
        if nthreads is None:
            nthreads = multiprocessing.cpu_count()
        firstrow = np.asarray(firstrow, dtype)
        if firstrow.shape[-1] == 1:
            self.__class__ = DiagonalOperator
            self.__init__(
                firstrow[..., 0], broadcast='rightward', shapein=shapein, **keywords
            )
            return
        nsamples = shapein[-1]
        bandwidth = 2 * firstrow.shape[-1] - 1
        ncorr = firstrow.shape[-1] - 1
        fftsize = 2
        while fftsize < nsamples + ncorr:
            fftsize *= 2
        with _pool.get(fftsize, dtype, aligned=True, contiguous=True) as rbuffer:
            with _pool.get(
                fftsize // 2 + 1, complex_dtype(dtype), aligned=True, contiguous=True
            ) as cbuffer:
                fplan = pyfftw.FFTW(
                    rbuffer, cbuffer, fftw_flags=[fftw_flag], threads=nthreads
                )
                bplan = pyfftw.FFTW(
                    cbuffer,
                    rbuffer,
                    direction='FFTW_BACKWARD',
                    fftw_flags=[fftw_flag],
                    threads=nthreads,
                )
                kernel = self._get_kernel(
                    firstrow, fplan, rbuffer, cbuffer, ncorr, fftsize, dtype
                )
        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)
        self.nsamples = nsamples
        self.fftsize = fftsize
        self.bandwidth = bandwidth
        self.ncorr = ncorr
        self.fplan = fplan
        self.bplan = bplan
        self.kernel = kernel
        self.fftw_flag = fftw_flag
        self.nthreads = nthreads

    def direct(self, x, out):
        with _pool.get(
            self.fftsize, self.dtype, aligned=True, contiguous=True
        ) as rbuffer:
            with _pool.get(
                self.fftsize // 2 + 1,
                complex_dtype(self.dtype),
                aligned=True,
                contiguous=True,
            ) as cbuffer:
                lpad = (self.bandwidth - 1) // 2
                x = x.reshape((-1, self.nsamples))
                out = out.reshape((-1, self.nsamples))
                self.fplan.update_arrays(rbuffer, cbuffer)
                self.bplan.update_arrays(cbuffer, rbuffer)

                for x_, out_, kernel in izip_broadcast(x, out, self.kernel):
                    rbuffer[:lpad] = 0
                    rbuffer[lpad : lpad + self.nsamples] = x_
                    rbuffer[lpad + self.nsamples :] = 0
                    self.fplan.execute()
                    cbuffer *= kernel
                    self.bplan.execute()
                    out_[...] = rbuffer[lpad : lpad + self.nsamples]

    def _get_kernel(self, firstrow, fplan, rbuffer, cbuffer, ncorr, fftsize, dtype):
        firstrow = firstrow.reshape((-1, ncorr + 1))
        kernel = empty((firstrow.shape[0], fftsize // 2 + 1), dtype)
        for f, k in izip(firstrow, kernel):
            rbuffer[: ncorr + 1] = f
            rbuffer[ncorr + 1 : -ncorr] = 0
            rbuffer[-ncorr:] = f[:0:-1]
            fplan.execute()
            k[...] = cbuffer.real / fftsize
        return kernel


@real
@linear
class DifferenceOperator(Operator):
    """
    Non-optimised difference operator.

    """

    def __init__(self, axis=-1, **keywords):
        self.axis = axis
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        output[...] = np.diff(input, axis=self.axis)

    def transpose(self, input, output):
        slices = [slice(None)] * input.ndim
        slices[self.axis] = slice(1, -1)
        shapetmp = list(input.shape)
        shapetmp[self.axis] += 2
        tmp = np.zeros(shapetmp)
        tmp[slices] = input
        output[...] = -np.diff(tmp, axis=self.axis)

    def reshapein(self, shapein):
        shape = list(shapein)
        shape[self.axis] -= 1
        return tuple(shape)

    def reshapeout(self, shapeout):
        shape = list(shapeout)
        shape[self.axis] += 1
        return tuple(shape)


@symmetric
class EigendecompositionOperator(CompositionOperator):
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
        W = DiagonalOperator(w)
        V = DenseOperator(v)
        V.set_rule('.T.', '1', CompositionOperator)
        self.eigenvalues = w
        self.eigenvectors = v
        CompositionOperator.__init__(self, [V, W, V.T], **kwargs)
        self.set_rule('.I', lambda s: s**-1)

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
        return EigendecompositionOperator(v=self.eigenvectors, w=self.eigenvalues**n)

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
