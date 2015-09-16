from __future__ import absolute_import, division, print_function

import numpy as np
import os
import time

from .config import LOCAL_PATH
from .core import (
    AdditionOperator, CompositionOperator, DiagonalOperator, HomothetyOperator,
    Operator, _pool)
from .flags import aligned, contiguous, inplace, linear, real, square, unitary
from .memory import empty
from .utils import (complex_dtype, isalias, omp_num_threads, product,
                    tointtuple)
from .utils.ufuncs import multiply_conjugate
from .warnings import warn, PyOperatorsWarning

try:
    import pyfftw
    FFTW_DEFAULT_NUM_THREADS = omp_num_threads()
    FFTW_WISDOM_FILES = tuple(os.path.join(LOCAL_PATH, 'fftw{0}.wisdom'.format(
                              t)) for t in ['', 'f', 'l'])
    FFTW_WISDOM_MIN_DELAY = 0.1
    _is_fftw_wisdom_loaded = False
except:
    warn('The pyFFTW library is not installed.', PyOperatorsWarning)

__all__ = ['ConvolutionOperator', 'FFTOperator']


# doctest nose fixture
def setup_module(module):
    try:
        import pyfftw
    except ImportError:
        from nose.plugins.skip import SkipTest
        raise SkipTest()


# FFTW out-of-place transforms:
# PRESERVE_INPUT: default except c2r and hc2r
# DESTROY_INPUT: default for c2r and hc2r, only possibility for multi c2r

OPERATOR_ATTRIBUTES = ['attrin', 'attrout', 'classin', 'classout', 'commin',
                       'commout', 'reshapein', 'reshapeout', 'shapein',
                       'shapeout', 'toshapein', 'toshapeout', 'validatein',
                       'validateout', 'dtype', 'flags']


@linear
@square
@inplace
@aligned
@contiguous
class _FFTWConvolutionOperator(Operator):
    """
    Multi-dimensional convolution by a real or complex kernel,
    using the discrete Fourier transform.

    """
    def __init__(self, kernel, shapein, axes=None, fftw_flag='FFTW_MEASURE',
                 nthreads=None, dtype=None, **keywords):
        """
        Parameters
        ----------
        kernel : array-like
            The multi-dimensional convolution kernel.
        shapein : tuple
            The shape of the input to be convolved by the kernel.
        axes : tuple
            Axes along which the convolution is performed. Convolution over
            less axes than the operator's input is not yet supported.
        fftw_flag : string
            list of strings and is a subset of the flags that FFTW allows for
            the planners. Specifically, FFTW_ESTIMATE, FFTW_MEASURE,
            FFTW_PATIENT and FFTW_EXHAUSTIVE are supported. These describe the
            increasing amount of effort spent during the planning stage to
            create the fastest possible transform. Usually, FFTW_MEASURE is
            a good compromise and is the default.
        nthreads : int
            Tells how many threads to use when invoking FFTW or MKL. Default is
            the number of cores.
        dtype : dtype
            Operator's dtype.

        """
        kernel = np.array(kernel, dtype=dtype, copy=False)
        dtype = kernel.dtype
        if dtype.kind not in ('f', 'c'):
            kernel = kernel.astype(float)
            dtype = kernel.dtype

        if shapein is None:
            raise ValueError('The input shape is not specified.')

        shapein = tointtuple(shapein)
        if len(shapein) != kernel.ndim:
            raise ValueError(
                "The kernel dimension '{0}' is incompatible with that of the s"
                "pecified shape '{1}'.".format(kernel.ndim, len(shapein)))

        # if the kernel is larger than the image, we don't crop it since it
        # might affect normalisation of the kernel
        if any([ks > s for ks, s in zip(kernel.shape, shapein)]):
            raise ValueError('The kernel must not be larger than the input.')

        if axes is None:
            axes = range(len(shapein))
        axes = tointtuple(axes)
        nthreads = nthreads or FFTW_DEFAULT_NUM_THREADS
        fftw_flag = fftw_flag.upper()

        if dtype.kind == 'c':
            n = product(shapein)
            fft = _FFTWComplexForwardOperator(shapein, axes, fftw_flag,
                                              nthreads, dtype, **keywords)
            kernel_fft = _get_kernel_fft(kernel, shapein, dtype, shapein,
                                         dtype, fft.oplan)
            kernel_fft /= n
            self.__class__ = CompositionOperator
            self.__init__([n, fft.H, DiagonalOperator(kernel_fft), fft])
            return

        dtype_ = complex_dtype(dtype)
        shape_ = self._reshape_to_halfstorage(shapein, axes)
        _load_wisdom()
        aligned = self.flags.aligned_input
        contiguous = True
        with _pool.get(shapein, dtype, aligned, contiguous) as in_:
            with _pool.get(shape_, dtype_, aligned, contiguous) as out:
                t0 = time.time()
                fplan = pyfftw.FFTW(in_, out, axes=axes,
                                    flags=[fftw_flag],
                                    direction='FFTW_FORWARD',
                                    threads=nthreads)
                bplan = pyfftw.FFTW(out, in_, axes=axes,
                                    flags=[fftw_flag],
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        if time.time() - t0 > FFTW_WISDOM_MIN_DELAY:
            _save_wisdom()

        kernel_fft = _get_kernel_fft(kernel, shapein, dtype, shape_,
                                     dtype_, fplan)
        kernel_fft /= product(shapein)
        self.__class__ = _FFTWRealConvolutionOperator
        self.__init__(kernel_fft, fplan, bplan, axes, fftw_flag, nthreads,
                      shapein, dtype, **keywords)

    def _reshape_to_halfstorage(self, shape, axes):
        shape = list(shape)
        shape[axes[-1]] = shape[axes[-1]] // 2 + 1
        return shape


@real
@linear
@square
@inplace
@aligned
@contiguous
class _FFTWRealConvolutionOperator(Operator):
    """
    Convolution by a real kernel.
    The first argument is the FFT of the real kernel. It is not necessarily
    aligned.

    """
    def __init__(self, kernel_fft, fplan, bplan, axes, fftw_flag, nthreads,
                 shapein=None, dtype=None, **keywords):
        self.kernel = kernel_fft
        self._fplan = fplan
        self._bplan = bplan
        self.axes = axes
        self.nthreads = nthreads
        self.fftw_flag = fftw_flag

        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)
        self.set_rule('T', lambda s: _FFTWRealConvolutionTransposeOperator(
            s.kernel, s._fplan, s._bplan, s.axes, s.fftw_flag, s.nthreads))
        self.set_rule(('.', HomothetyOperator), self._rule_homothety,
                      CompositionOperator)
        self.set_rule(('.', _FFTWRealConvolutionOperator), self.
                      _rule_add_real, AdditionOperator)
        self.set_rule(('.', _FFTWRealConvolutionOperator), self.
                      _rule_cmp_real, CompositionOperator)
        self.set_rule(('.', _FFTWComplexBackwardOperator), self.
                      _rule_complex_backward, CompositionOperator)
        self.set_rule((_FFTWComplexForwardOperator, '.'), self.
                      _rule_complex_forward, CompositionOperator)

    def direct(self, input, output):
        shape = self.kernel.shape
        dtype = self.kernel.dtype
        aligned = self.flags.aligned_input
        contiguous = True
        with _pool.get(shape, dtype, aligned, contiguous) as buf:
            self._fplan.update_arrays(input, buf)
            self._fplan.execute()
            buf *= self.kernel
            self._bplan.update_arrays(buf, output)
            self._bplan.execute()

    def get_kernel(self, out=None):
        if out is not None:
            out[...] = self.kernel
        return self.kernel

    @property
    def nbytes(self):
        return self.kernel.nbytes

    @staticmethod
    def _rule_homothety(self, scalar):
        kernel = empty(self.kernel.shape, self.kernel.dtype)
        self.get_kernel(kernel)
        kernel *= scalar.data
        result = _FFTWRealConvolutionOperator(
            kernel, self._fplan, self._bplan, self.axes, self.fftw_flag,
            self.nthreads, self.shapein, self.dtype)
        return result

    @staticmethod
    def _rule_add_real(self, other):
        if isinstance(other, _FFTWRealConvolutionTransposeOperator):
            # spare allocation in other.get_kernel (if self is not a transpose)
            self, other = other, self
        kernel = empty(self.kernel.shape, self.kernel.dtype)
        self.get_kernel(kernel)
        np.add(kernel, other.get_kernel(), kernel)
        result = _FFTWRealConvolutionOperator(
            kernel, self._fplan, self._bplan, self.axes, self.fftw_flag,
            self.nthreads, self.shapein, self.dtype)
        return result

    @staticmethod
    def _rule_cmp_real(self, other):
        if isinstance(other, _FFTWRealConvolutionTransposeOperator):
            # spare allocation in other.get_kernel (if self is not a transpose)
            self, other = other, self
        kernel = empty(self.kernel.shape, self.kernel.dtype)
        self.get_kernel(kernel)
        kernel *= other.get_kernel()
        kernel *= product(self.shapein)
        result = _FFTWRealConvolutionOperator(
            kernel, self._fplan, self._bplan, self.axes, self.fftw_flag,
            self.nthreads, self.shapein, self.dtype)
        return result

    @staticmethod
    def _rule_complex_backward(self, other):
        kernel = self._restore_kernel().astype(self.kernel.dtype)
        other.H.direct(kernel, kernel)
        kernel /= product(self.shapein)
        return other, DiagonalOperator(kernel)

    @staticmethod
    def _rule_complex_forward(other, self):
        kernel = self._restore_kernel().astype(self.kernel.dtype)
        other.direct(kernel, kernel)
        return DiagonalOperator(kernel), other

    def _restore_kernel(self):
        shape = self.kernel.shape
        dtype = self.kernel.dtype
        aligned = self.flags.aligned_input
        contiguous = True
        with _pool.get(shape, dtype, aligned, contiguous) as x:
            self.get_kernel(x)
            y = empty(self.shapein, self.dtype)
            self._bplan.update_arrays(x, y)
        self._bplan.execute()
        return y


class _FFTWRealConvolutionTransposeOperator(_FFTWRealConvolutionOperator):
    """
    Transpose of the convolution by a real kernel.

    """
    __name__ = '_FFTW_RealConvolutionOperator.T'

    def get_kernel(self, out=None):
        return np.conjugate(self.kernel, out)

    def direct(self, input, output):
        with _pool.get(self.kernel.shape, self.kernel.dtype) as buf:
            self._fplan.update_arrays(input, buf)
            self._fplan.execute()
            multiply_conjugate(buf, self.kernel, buf)
            self._bplan.update_arrays(buf, output)
            self._bplan.execute()


@linear
@square
@inplace
@aligned
@contiguous
class _FFTWComplexOperator(Operator):
    def __init__(self, shapein, axes=None, fftw_flag='FFTW_MEASURE',
                 nthreads=None, dtype=complex, **keywords):
        shapein = tointtuple(shapein)
        if axes is None:
            axes = range(len(shapein))
        self.axes = tointtuple(axes)
        self.fftw_flag = fftw_flag.upper()
        self.nthreads = nthreads or FFTW_DEFAULT_NUM_THREADS
        dtype = np.dtype(dtype)
        _load_wisdom()
        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)

    def direct(self, input, output):
        if isalias(input, output):
            self.iplan.update_arrays(input, output)
            self.iplan.execute()
        else:
            self.oplan.update_arrays(input, output)
            self.oplan.execute()


@unitary
class _FFTWComplexForwardOperator(_FFTWComplexOperator):
    """
    Complex multi-dimensional forward Discrete Fourier Transform.

    """
    def __init__(self, shapein, axes=None, fftw_flag='FFTW_MEASURE',
                 nthreads=None, dtype=complex, **keywords):
        """
        Parameters
        ----------
        shapein : tuple
            The shape of the input to be Fourier-transformed
        axes : tuple
            Axes along which the transform is performed.
        fftw_flag : string
            FFTW flag for the planner: FFTW_ESTIMATE, FFTW_MEASURE,
            FFTW_PATIENT or FFTW_EXHAUSTIVE. These describe the
            increasing amount of effort spent during the planning stage to
            create the fastest possible transform. Usually, FFTW_MEASURE is
            a good compromise and is the default.
        nthreads : int
            Tells how many threads to use when invoking FFTW or MKL. Default is
            the number of cores.
        dtype : dtype
            Operator's complex dtype.

        """
        _FFTWComplexOperator.__init__(self, shapein, axes, fftw_flag,
                                      nthreads, dtype, **keywords)

        self.set_rule('H', lambda s:
                      HomothetyOperator(1 / product(s.shapein)) *
                      _FFTWComplexBackwardOperator(s.shapein, forward=s))
        self.set_rule((_FFTWComplexBackwardOperator, '.'), lambda o, s:
                      HomothetyOperator(product(s.shapein)),
                      CompositionOperator)

        with _pool.get(shapein, dtype) as in_:
            t0 = time.time()
            self.iplan = pyfftw.FFTW(in_, in_, axes=self.axes,
                                     flags=[self.fftw_flag],
                                     direction='FFTW_FORWARD',
                                     threads=self.nthreads)
            with _pool.get(shapein, dtype) as out:
                self.oplan = pyfftw.FFTW(in_, out, axes=self.axes,
                                         flags=[self.fftw_flag],
                                         direction='FFTW_FORWARD',
                                         threads=self.nthreads)
        if time.time() - t0 > FFTW_WISDOM_MIN_DELAY:
            _save_wisdom()


class _FFTWComplexBackwardOperator(_FFTWComplexOperator):
    """
    Complex multi-dimensional backward Discrete Fourier Transform.

    """
    def __init__(self, shapein, dtype=None, forward=None, **keywords):

        dtype = dtype or forward.dtype
        _FFTWComplexOperator.__init__(self, shapein, forward.axes,
                                      forward.fftw_flag,
                                      forward.nthreads, dtype, **keywords)
        self.set_rule('H', lambda s:
                      HomothetyOperator(product(s.shapein)) * forward)
        self.set_rule((_FFTWComplexForwardOperator, '.'), lambda o, s:
                      HomothetyOperator(product(s.shapein)),
                      CompositionOperator)

        with _pool.get(shapein, dtype) as in_:
            t0 = time.time()
            self.iplan = pyfftw.FFTW(in_, in_, axes=self.axes,
                                     flags=[self.fftw_flag],
                                     direction='FFTW_BACKWARD',
                                     threads=self.nthreads)
            with _pool.get(shapein, dtype) as out:
                self.oplan = pyfftw.FFTW(in_, out, axes=self.axes,
                                         flags=[self.fftw_flag],
                                         direction='FFTW_BACKWARD',
                                         threads=self.nthreads)
        if time.time() - t0 > FFTW_WISDOM_MIN_DELAY:
            _save_wisdom()


def _get_kernel_fft(kernel, shapein, dtypein, shapeout, dtypeout, fft):
    with _pool.get(shapein, dtypein) as kernel_padded:
        ker_slice = [slice(0, s) for s in kernel.shape]
        kernel_padded[...] = 0
        kernel_padded[ker_slice] = kernel
        ker_origin = (np.array(kernel.shape)-1) // 2
        for axis, o in enumerate(ker_origin):
            kernel_padded = np.roll(kernel_padded, int(-o), axis=axis)
        kernel_fft = empty(shapeout, dtypeout)
        fft.update_arrays(kernel_padded, kernel_fft)
        fft.execute()
        return kernel_fft


def _load_wisdom():
    """ Loads the 3 wisdom files. """
    global _is_fftw_wisdom_loaded
    if _is_fftw_wisdom_loaded:
        return

    def load(filename):
        try:
            with open(filename, 'rb') as f:
                wisdom = f.read()
        except IOError:
            wisdom = b''
        return wisdom

    wisdom = [load(f) for f in FFTW_WISDOM_FILES]
    pyfftw.import_wisdom(wisdom)
    _is_fftw_wisdom_loaded = True


def _save_wisdom():
    """ Save wisdom as 3 files. """
    wisdom = pyfftw.export_wisdom()
    for filename, w in zip(FFTW_WISDOM_FILES, wisdom):
        try:
            os.remove(filename)
        except OSError:
            pass
        if len(w) == 0:
            continue
        with open(filename, 'wb') as f:
            f.write(w)


# make FFTW the default
ConvolutionOperator = _FFTWConvolutionOperator
FFTOperator = _FFTWComplexForwardOperator
