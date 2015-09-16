#coding: utf-8
from __future__ import absolute_import, division, print_function
import numexpr
if numexpr.__version__ < '2.0':
    raise ImportError('Please update numexpr to a newer version >= 2.0.')

import numpy as np
import pyoperators as po
from .core import (
    BlockColumnOperator, CompositionOperator, ConstantOperator, DiagonalBase,
    IdentityOperator, MultiplicationOperator, Operator, ReductionOperator)
from .flags import (
    idempotent, inplace, real, separable, square, update_output)
from .utils import (
    operation_assignment, operation_symbol, pi, strenum, tointtuple)
from .utils.ufuncs import hard_thresholding, soft_thresholding

__all__ = ['Cartesian2SphericalOperator',
           'ClipOperator',
           'HardThresholdingOperator',
           'MaxOperator',
           'MinOperator',
           'MinMaxOperator',
           'MaximumOperator',
           'MinimumOperator',
           'NormalizeOperator',
           'NumexprOperator',
           'PowerOperator',
           'ProductOperator',
           'ReciprocalOperator',
           'RoundOperator',
           'SoftThresholdingOperator',
           'Spherical2CartesianOperator',
           'SqrtOperator',
           'SquareOperator',
           'To1dOperator',
           'ToNdOperator']


@real
class _CartesianSpherical(Operator):
    CONVENTIONS = ('zenith,azimuth',
                   'azimuth,zenith',
                   'elevation,azimuth',
                   'azimuth,elevation')

    def __init__(self, convention, dtype=float, **keywords):
        if not isinstance(convention, str):
            raise TypeError("The input convention '{0}' is not a string.".
                            format(convention))
        convention_ = convention.replace(' ', '').lower()
        if convention_ not in self.CONVENTIONS:
            raise ValueError(
                "Invalid spherical convention '{0}'. Expected values are {1}.".
                format(convention, strenum(self.CONVENTIONS)))
        self.convention = convention_
        Operator.__init__(self, dtype=dtype, **keywords)

    @staticmethod
    def _reshapecartesian(shape):
        return shape[:-1] + (2,)

    @staticmethod
    def _reshapespherical(shape):
        return shape[:-1] + (3,)

    @staticmethod
    def _validatecartesian(shape):
        if len(shape) == 0 or shape[-1] != 3:
            raise ValueError('Invalid cartesian shape.')

    @staticmethod
    def _validatespherical(shape):
        if len(shape) == 0 or shape[-1] != 2:
            raise ValueError('Invalid spherical shape.')

    @staticmethod
    def _rule_identity(s, o):
        if s.convention == o.convention:
            return IdentityOperator()


class Cartesian2SphericalOperator(_CartesianSpherical):
    """
    Convert cartesian unit vectors into spherical coordinates in radians
    or degrees.

    The spherical coordinate system is defined by:
       - the zenith direction of coordinate (0, 0, 1)
       - the azimuthal reference of coordinate (1, 0, 0)
       - the azimuth signedness: it is counted positively from the X axis
    to the Y axis.

    The last dimension of the operator's output is 2 and it encodes
    the two spherical angles. Four conventions define what these angles are:
       - 'zenith,azimuth': (theta, phi) angles commonly used
       in physics or the (colatitude, longitude) angles used
       in the celestial and geographical coordinate systems
       - 'azimuth,zenith': (longitude, colatitude) convention
       - 'elevation,azimuth: (latitude, longitude) convention
       - 'azimuth,elevation': (longitude, latitude) convention

    """
    def __init__(self, convention, degrees=False, **keywords):
        """
        convention : string
            One of the following spherical coordinate conventions:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.
        degrees : boolean, optional
            If true, the spherical coordinates are returned in degrees.

        """
        if degrees:
            self.__class__ = CompositionOperator
            self.__init__(
                [po.linear.DegreesOperator(),
                 Cartesian2SphericalOperator(convention, **keywords)])
            return
        self.degrees = False

        _CartesianSpherical.__init__(
            self, convention,
            reshapein=self._reshapecartesian,
            reshapeout=self._reshapespherical,
            validatein=self._validatecartesian,
            validateout=self._validatespherical,
            **keywords)
        self.set_rule('I',
                      lambda s: Spherical2CartesianOperator(s.convention))
        self.set_rule(('.', Spherical2CartesianOperator), self._rule_identity,
                      CompositionOperator)

    def direct(self, input, output):
        if self.convention.startswith('azimuth'):
            o1, o2 = output[..., 1], output[..., 0]
        else:
            o1, o2 = output[..., 0], output[..., 1]
        np.arccos(input[..., 2], o1)
        if 'elevation' in self.convention:
            np.subtract(pi(self.dtype) / 2, o1, o1)
        np.arctan2(input[..., 1], input[..., 0], o2)
        if o2.ndim > 0:
            o2[o2 < 0] += 2 * pi(self.dtype)
        elif o2 < 0:
            o2 += 2 * pi(self.dtype)


class Spherical2CartesianOperator(_CartesianSpherical):
    """
    Convert spherical coordinates in radians or degrees into unit cartesian
    vectors.

    The spherical coordinate system is defined by:
       - the zenith direction of coordinate (0, 0, 1)
       - the azimuthal reference of coordinate (1, 0, 0)
       - the azimuth signedness: it is counted positively from the X axis
    to the Y axis.

    The last dimension of the operator's input is 2 and it encodes
    the two spherical angles. Four conventions define what these angles are:
       - 'zenith,azimuth': (theta, phi) angles commonly used
       in physics or the (colatitude, longitude) angles used
       in the celestial and geographical coordinate systems
       - 'azimuth,zenith': (longitude, colatitude) convention
       - 'elevation,azimuth: (latitude, longitude) convention
       - 'azimuth,elevation': (longitude, latitude) convention

    """
    def __init__(self, convention, degrees=False, **keywords):
        """
        convention : string
            One of the following spherical coordinate conventions:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.
        degrees : boolean, optional
            If true, the input spherical coordinates are assumed to be in
            degrees.

        """
        if degrees:
            self.__class__ = CompositionOperator
            self.__init__(
                [Spherical2CartesianOperator(convention, **keywords),
                 po.linear.RadiansOperator()])
            return
        self.degrees = False

        _CartesianSpherical.__init__(
            self, convention,
            reshapein=self._reshapespherical,
            reshapeout=self._reshapecartesian,
            validatein=self._validatespherical,
            validateout=self._validatecartesian,
            **keywords)
        self.set_rule('I',
                      lambda s: Cartesian2SphericalOperator(s.convention))
        self.set_rule(('.', Cartesian2SphericalOperator), self._rule_identity,
                      CompositionOperator)

    def direct(self, input, output):
        if self.convention.startswith('azimuth'):
            theta, phi = input[..., 1], input[..., 0]
        else:
            theta, phi = input[..., 0], input[..., 1]
        if 'elevation' in self.convention:
            theta = 0.5 * pi(self.dtype) - theta
        sintheta = np.sin(theta)
        np.multiply(sintheta, np.cos(phi), output[..., 0])
        np.multiply(sintheta, np.sin(phi), output[..., 1])
        np.cos(theta, output[..., 2])


@square
@inplace
@separable
class ClipOperator(Operator):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Arguments
    ---------
    minvalue: scalar or array_like
        The minimum limit below which all input values are set to vmin.
    maxvalue: scalar or array_like
        The maximum limit above which all input values are set to vmax.

    Exemples
    --------
    >>> C = ClipOperator(0, 1)
    >>> x = np.linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> C(x)
    array([ 0.,  0.,  0.,  1.,  1.])

    See also
    --------
    MaximumOperator, MinimumOperator, np.clip

    """
    def __init__(self, minvalue, maxvalue, **keywords):
        self.minvalue = np.asarray(minvalue)
        self.maxvalue = np.asarray(maxvalue)
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        np.clip(input, self.minvalue, self.maxvalue, out=output)

    @property
    def nbytes(self):
        return self.minvalue.nbytes + self.maxvalue.nbytes

    def __str__(self):
        return 'clip(..., {0}, {1})'.format(self.minvalue, self.maxvalue)


@square
@inplace
@separable
class PowerOperator(Operator):
    'X -> X**n'
    def __init__(self, n, dtype=float, **keywords):
        if np.allclose(n, -1) and not isinstance(self, ReciprocalOperator):
            self.__class__ = ReciprocalOperator
            self.__init__(dtype=dtype, **keywords)
            return
        if n == 0:
            self.__class__ = ConstantOperator
            self.__init__(1, dtype=dtype, **keywords)
            return
        if np.allclose(n, 0.5) and not isinstance(self, SqrtOperator):
            self.__class__ = SqrtOperator
            self.__init__(dtype=dtype, **keywords)
            return
        if np.allclose(n, 1):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        if np.allclose(n, 2) and not isinstance(self, SquareOperator):
            self.__class__ = SquareOperator
            self.__init__(dtype=dtype, **keywords)
            return
        self.n = n
        Operator.__init__(self, dtype=dtype, **keywords)
        self.set_rule('I', lambda s: PowerOperator(1/s.n))
        self.set_rule(('.', PowerOperator),
                      lambda s, o: PowerOperator(s.n * o.n),
                      CompositionOperator)
        self.set_rule(('.', PowerOperator),
                      lambda s, o: PowerOperator(s.n + o.n),
                      MultiplicationOperator)
        self.set_rule(('.', DiagonalBase),
                      lambda s, o: MultiplicationOperator(
                          [ConstantOperator(o.get_data(),
                                            broadcast=o.broadcast),
                           PowerOperator(s.n + 1)]),
                      MultiplicationOperator)

    def direct(self, input, output):
        np.power(input, self.n, output)

    @property
    def nbytes(self):
        return self.n.nbytes

    def __str__(self):
        return '...**{0}'.format(self.n)


class ReciprocalOperator(PowerOperator):
    'X -> 1 / X'
    def __init__(self, **keywords):
        PowerOperator.__init__(self, -1, **keywords)

    def direct(self, input, output):
        np.reciprocal(input, output)

    def __str__(self):
        return '1/...'


class SqrtOperator(PowerOperator):
    'X -> sqrt(X)'
    def __init__(self, **keywords):
        PowerOperator.__init__(self, 0.5, **keywords)

    def direct(self, input, output):
        np.sqrt(input, output)


class SquareOperator(PowerOperator):
    'X -> X**2'
    def __init__(self, **keywords):
        PowerOperator.__init__(self, 2, **keywords)

    def direct(self, input, output):
        np.square(input, output)

    def __str__(self):
        return u'...²'.encode('utf-8')


class ProductOperator(ReductionOperator):
    """
    Product-along-axis operator.

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
    >>> op = ProductOperator()
    >>> op([1,2,3])
    array(6)

    """
    def __init__(self, axis=None, dtype=None, skipna=True, **keywords):
        ReductionOperator.__init__(self, np.multiply, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)

    def __str__(self):
        return 'product' if self.axis is None \
                         else 'product(..., axis={0})'.format(self.axis)


class MaxOperator(ReductionOperator):
    """
    Max-along-axis operator.

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
    >>> op = MaxOperator()
    >>> op([1,2,3])
    array(3)

    """
    def __init__(self, axis=None, dtype=None, skipna=False, **keywords):
        if np.__version__ < '2':
            func = np.nanmax if skipna else np.max
        else:
            func = np.max
        ReductionOperator.__init__(self, func, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)

    def __str__(self):
        return 'max' if self.axis is None \
                     else 'max(..., axis={0})'.format(self.axis)


class MinOperator(ReductionOperator):
    """
    Min-along-axis operator.

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
    >>> op = MinOperator()
    >>> op([1,2,3])
    array(1)

    """
    def __init__(self, axis=None, dtype=None, skipna=False, **keywords):
        if np.__version__ < '2':
            func = np.nanmin if skipna else np.min
        else:
            func = np.min
        ReductionOperator.__init__(self, func, axis=axis, dtype=dtype,
                                   skipna=skipna, **keywords)

    def __str__(self):
        return 'min' if self.axis is None \
                     else 'min(..., axis={0})'.format(self.axis)


class MinMaxOperator(BlockColumnOperator):
    """
    MinMax-along-axis operator.

    Parameters
    ----------
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    new_axisout : integer, optional
        Axis in which the minimum and maximum values are set.
    dtype : dtype, optional
        Operator data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = MinMaxOperator()
    >>> op([1,2,3])
    array([1, 3])
    >>> op = MinMaxOperator(axis=0, new_axisout=0)
    >>> op([[1,2,3],[2,1,4],[0,1,8]])
    array([[0, 1, 3],
           [2, 2, 8]])

    """
    def __init__(self, axis=None, dtype=None, skipna=False, new_axisout=-1,
                 **keywords):
        operands = [MinOperator(axis=axis, dtype=dtype, skipna=skipna),
                    MaxOperator(axis=axis, dtype=dtype, skipna=skipna)]
        BlockColumnOperator.__init__(self, operands, new_axisout=new_axisout,
                                     **keywords)

    def __str__(self):
        return 'minmax' if self.axis is None \
                        else 'minmax(..., axis={0})'.format(self.axis)


@square
@inplace
@separable
class MaximumOperator(Operator):
    """
    Set all input array values above a given value to this value.

    Arguments
    ---------
    value: scalar or array_like
        Threshold value to which the input array is compared.

    Exemple
    -------
    >>> M = MaximumOperator(1)
    >>> x = np.linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> M(x)
    array([ 1.,  1.,  1.,  1.,  2.])

    See also
    --------
    ClipOperator, MinimumOperator, np.maximum

    """
    def __init__(self, value, **keywords):
        self.value = np.asarray(value)
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        np.maximum(input, self.value, output)

    @property
    def nbytes(self):
        return self.value.nbytes

    def __str__(self):
        return 'maximum(..., {0})'.format(self.value)


@square
@inplace
@separable
class MinimumOperator(Operator):
    """
    Set all input array values above a given value to this value.

    Arguments
    ---------
    value: scalar, broadcastable array
        The value to which the input array is compared.

    Exemple
    -------
    >>> M = MinimumOperator(1)
    >>> x = np.linspace(-2, 2, 5)
    >>> x
    array([-2., -1.,  0.,  1.,  2.])
    >>> M(x)
    array([-2., -1.,  0.,  1.,  1.])

    See also
    --------
    ClipOperator, MaximumOperator, np.minimum

    """
    def __init__(self, value, **keywords):
        self.value = np.asarray(value)
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        np.minimum(input, self.value, output)

    @property
    def nbytes(self):
        return self.value.nbytes

    def __str__(self):
        return 'minimum(..., {0})'.format(self.value)


@square
@inplace
class NormalizeOperator(Operator):
    """
    Normalize a cartesian vector.

    Example
    -------
    >>> n = NormalizeOperator()
    >>> n([1, 1])
    array([ 0.70710678,  0.70710678])

    """
    def __init__(self, dtype=float, **keywords):
        Operator.__init__(self, dtype=dtype, **keywords)

    def direct(self, input, output):
        np.divide(input, np.sqrt(np.sum(input**2, axis=-1))[..., None], output)


@square
@inplace
@update_output
class NumexprOperator(Operator):
    """
    Return an operator evaluating an expression using numexpr.

    Parameters
    ----------
    expr : string
        The numexp expression to be evaluated. It must contain the 'input'
        variable name.
    global_dict : dict
        A dictionary of global variables that are passed to numexpr's
        'evaluate' method.

    Example
    -------
    >>> k = 1.2
    >>> op = NumexprOperator('exp(input+k)', {'k':k})
    >>> print(op(1) == np.exp(2.2))
    True

    """
    def __init__(self, expr, global_dict=None, dtype=float, **keywords):
        self.expr = expr
        self.global_dict = global_dict
        if numexpr.__version__ < '2.1':
            keywords['flags'] = self.validate_flags(
                keywords.get('flags', {}), update_output=False)
        Operator.__init__(self, dtype=dtype, **keywords)

    def direct(self, input, output, operation=operation_assignment):
        if operation is operation_assignment:
            expr = self.expr
        else:
            op = operation_symbol[operation]
            expr = 'output' + op + '(' + self.expr + ')'
        numexpr.evaluate(expr, global_dict=self.global_dict, out=output)

    @property
    def nbytes(self):
        if self.global_dict is None:
            return 0
        return np.sum(v.nbytes for v in self.global_dict.values()
                               if hasattr(v, 'nbytes'))

    def __str__(self):
        return 'numexpr({0}, ...)'.format(self.expr)


@square
@idempotent
@inplace
@separable
class RoundOperator(Operator):
    """
    Rounding operator.

    The rounding method may be one of the following:
        - rtz : round towards zero (truncation)
        - rti : round towards infinity (Not implemented)
        - rtmi : round towards minus infinity (floor)
        - rtpi : round towards positive infinity (ceil)
        - rhtz : round half towards zero (Not implemented)
        - rhti : round half towards infinity (Fortran's nint)
        - rhtmi : round half towards minus infinity
        - rhtpi : round half towards positive infinity
        - rhte : round half to even (numpy's round),
        - rhto : round half to odd
        - rhs : round half stochastically (Not implemented)

    """
    def __init__(self, method='rhte', **keywords):
        method = method.lower()
        table = {'rtz': np.trunc,
                 #'rti'
                 'rtmi': np.floor,
                 'rtpi': np.ceil,
                 #'rhtz'
                 #'rhti'
                 'rhtmi': self._direct_rhtmi,
                 'rhtpi': self._direct_rhtpi,
                 'rhte': lambda i, o: np.round(i, 0, o),
                 #'rhs'
                 }
        if method not in table:
            raise ValueError(
                'Invalid rounding method. Expected values are {0}.'.format(
                strenum(table.keys())))
        Operator.__init__(self, table[method], **keywords)
        self.method = method

    @staticmethod
    def _direct_rhtmi(input, output):
        """ Round half to -inf. """
        np.add(input, 0.5, output)
        np.ceil(output, output)
        np.add(output, -1, output)

    @staticmethod
    def _direct_rhtpi(input, output):
        """ Round half to +inf. """
        np.add(input, -0.5, output)
        np.floor(output, output)
        np.add(output, 1, output)

    def __str__(self):
        method = self.method[1:]
        if method == 'rmi':
            method = 'floor'
        elif method == 'tpi':
            method = 'ceil'
        elif method == 'tz':
            method = 'trunc'
        return 'round_{0}'.format(method)


@square
@idempotent
@inplace
@separable
class HardThresholdingOperator(Operator):
    """
    Hard thresholding operator.

    Ha(x) = x if |x| > a,
            0 otherwise.

    Parameter
    ---------
    a : positive float or array
        The hard threshold.

    """
    def __init__(self, a, **keywords):
        a = np.asarray(a)
        if np.any(a < 0):
            raise ValueError('Negative hard threshold.')
        if a.ndim > 0:
            keywords['shapein'] = a.shape
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        if np.all(a == 0):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        Operator.__init__(self, **keywords)
        self.a = a
        self.set_rule(('.', HardThresholdingOperator), lambda s, o:
                      HardThresholdingOperator(np.maximum(s.a, o.a)),
                      CompositionOperator)

    def direct(self, input, output):
        hard_thresholding(input, self.a, output)

    @property
    def nbytes(self):
        return self.a.nbytes

    def __str__(self):
        return 'hardthreshold(..., {0})'.format(self.a)


@square
@inplace
@separable
class SoftThresholdingOperator(Operator):
    """
    Soft thresholding operator.

    Sa(x) = sign(x) [|x| - a]+

    Parameter
    ---------
    a : positive float or array
        The soft threshold.

    """
    def __init__(self, a, **keywords):
        a = np.asarray(a)
        if np.any(a < 0):
            raise ValueError('Negative soft threshold.')
        if a.ndim > 0:
            keywords['shapein'] = a.shape
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        if np.all(a == 0):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        Operator.__init__(self, **keywords)
        self.a = a

    def direct(self, input, output):
        soft_thresholding(input, self.a, output)

    @property
    def nbytes(self):
        return self.a.nbytes

    def __str__(self):
        return 'softthreshold(..., {0})'.format(self.a)


@separable
class _1dNdOperator(Operator):
    """ Base class for 1d-Nd coordinate mappings. """
    def __init__(self, shape_, order='C', **keywords):
        shape_ = tointtuple(shape_)
        ndim = len(shape_)
        if ndim == 1:
            raise NotImplementedError('ndim == 1 is not implemented.')
        if order.upper() not in ('C', 'F'):
            raise ValueError("Invalid order '{0}'. Expected order is 'C' or 'F"
                             "'".format(order))
        order = order.upper()

        Operator.__init__(self, **keywords)
        self.shape_ = shape_
        self.order = order
        self.ndim = ndim
        if order == 'C':
            self.coefs = np.cumproduct((1,) + shape_[:0:-1])[::-1]
        elif order == 'F':
            self.coefs = np.cumproduct((1,) + shape_[:-1])

    def _reshape_to1d(self, shape):
        return shape[:-1]

    def _reshape_tond(self, shape):
        return shape + (self.ndim,)

    def _validate_to1d(self, shape):
        if shape[-1] != self.ndim:
            raise ValueError("Invalid shape '{0}'. The expected last dimension"
                             " is '{1}'.".format(shape, self.ndim))


class To1dOperator(_1dNdOperator):
    """
    Convert an N-dimensional indexing to a 1-dimensional indexing.

    C order:
    -------------------------      -------------
    | (0,0) | (0,1) | (0,2) |      | 0 | 1 | 2 |
    -------------------------  =>  -------------
    | (1,0) | (1,1) | (1,2) |      | 3 | 4 | 5 |
    -------------------------      -------------

    Fortan order:
    -------------------------      -------------
    | (0,0) | (0,1) | (0,2) |      | 0 | 2 | 4 |
    -------------------------  =>  -------------
    | (1,0) | (1,1) | (1,2) |      | 1 | 3 | 5 |
    -------------------------      -------------

    Parameters
    ----------
    shape : tuple of int
        The shape of the array whose element' multi-dimensional coordinates
        will be converted into 1-d coordinates.
    order : str
        'C' for row-major and 'F' for column-major 1-d indexing.

    """
    def __init__(self, shape_, order='C', **keywords):
        if 'reshapein' not in keywords:
            keywords['reshapein'] = self._reshape_to1d
        if 'reshapeout' not in keywords:
            keywords['reshapeout'] = self._reshape_tond
        if 'validatein' not in keywords:
            keywords['validatein'] = self._validate_to1d
        _1dNdOperator.__init__(self, shape_, order=order, **keywords)
        self.set_rule('I', lambda s: ToNdOperator(s.shape_, order=s.order))

    def direct(self, input, output):
        np.dot(input, self.coefs, out=output)


class ToNdOperator(_1dNdOperator):
    """
    Convert a 1-dimensional indexing to an N-dimensional indexing.

    C order:
    -------------      -------------------------
    | 0 | 1 | 2 |      | (0,0) | (0,1) | (0,2) |
    -------------  =>  -------------------------
    | 3 | 4 | 5 |      | (1,0) | (1,1) | (1,2) |
    -------------      -------------------------

    Fortan order
    -------------      -------------------------
    | 0 | 2 | 4 |      | (0,0) | (0,1) | (0,2) |
    -------------  =>  -------------------------
    | 1 | 3 | 5 |      | (1,0) | (1,1) | (1,2) |
    -------------      -------------------------

    Parameters
    ----------
    shape : tuple of int
        The shape of the array whose element' multi-dimensional coordinates
        will be converted into 1-d coordinates.
    order : str
        'C' for row-major and 'F' for column-major 1-d indexing.

    """
    def __init__(self, shape_, order='C', **keywords):
        if 'reshapein' not in keywords:
            keywords['reshapein'] = self._reshape_tond
        if 'reshapeout' not in keywords:
            keywords['reshapeout'] = self._reshape_to1d
        if 'validateout' not in keywords:
            keywords['validateout'] = self._validate_to1d
        _1dNdOperator.__init__(self, shape_, order=order, **keywords)
        self.set_rule('I', lambda s: To1dOperator(
            s.shape_, order=s.order))

    def direct(self, input, output):
        np.floor_divide(input[..., None], self.coefs, out=output)
        np.mod(output, self.shape_, out=output)

    def __str__(self):
        return 'toNd'
