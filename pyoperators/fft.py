from __future__ import division, print_function

import numpy as np
import os
import time

from .config import LOCAL_PATH
from .core import (Operator, CompositionOperator, DiagonalOperator,
                   HomothetyOperator, ReverseOperatorFactory, _pool)
from .decorators import (aligned, contiguous, inplace, linear, real, square,
                         unitary)
from .memory import empty
from .utils import openmp_num_threads, product, tointtuple
from .utils.ufuncs import multiply_conjugate

__all__ = ['ConvolutionOperator', 'FFTOperator']

try:
    import pyfftw
    FFTW_ALIGNMENT = 16
    FFTW_DEFAULT_NUM_THREADS = openmp_num_threads()
    FFTW_WISDOM_FILES = tuple(os.path.join(LOCAL_PATH, 'fftw{0}.wisdom'.format(
                              t)) for t in ['', 'f', 'l'])
    FFTW_WISDOM_MIN_DELAY = 0.1
    _is_fftw_wisdom_loaded = False
except:
    print('Warning: The pyFFTW library is not installed.')

# FFTW out-of-place transforms:
# PRESERVE_INPUT: default except c2r and hc2r
# DESTROY_INPUT: default for c2r and hc2r, only possibility for multi c2r

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
    def __init__(self, kernel, shapein=None, dtype=None, axes=None,
                 fftw_flag='FFTW_MEASURE', nthreads=None, **keywords):
        """
        Parameters
        ----------
        kernel : array-like
            The multi-dimensional convolution kernel.
        shapein : tuple
            The shape of the input to be convolved by the kernel.
        dtype : dtype
            Operator's dtype.
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

        """
        kernel = np.array(kernel, dtype=dtype, copy=False)
        dtype = kernel.dtype
        if dtype.kind not in ('f', 'c'):
            kernel = kernel.astype(float)
            dtype = kernel.dtype

        if not isinstance(self, _FFTWRealConvolutionOperator) and \
           dtype.kind == 'f':
            self.__class__ = _FFTWRealConvolutionOperator
            self.__init__(kernel, shapein, axes, fftw_flag, nthreads,
                          **keywords)
            return
        if not isinstance(self, _FFTWComplexConvolutionOperator) and \
           dtype.kind == 'c':
            self.__class__ = _FFTWComplexConvolutionOperator
            self.__init__(kernel, shapein, axes, fftw_flag, nthreads,
                          **keywords)
            return

        if shapein is None:
            shapein = kernel.shape
        else:
            shapein = tointtuple(shapein)
        if len(shapein) != kernel.ndim:
            raise ValueError("The kernel dimension '{0}' is incompatible with t"
                "hat of the specified shape '{1}'.".format(kernel.ndim,
                len(shapein)))
        # if the kernel is larger than the image, we don't crop it since it
        # might affect normalisation of the kernel
        if any([ks > s for ks,s in zip(kernel.shape, shapein)]):
            raise ValueError('The kernel must not be larger than the input.')

        if axes is None:
            axes = range(len(shapein))
        self.axes = tointtuple(axes)
        self.nthreads = nthreads or FFTW_DEFAULT_NUM_THREADS
        self.fftw_flag = fftw_flag.upper()

        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)


class _FFTWComplexConvolutionOperator(_FFTWConvolutionOperator):
    """
    Convolution by a complex kernel.

    """
    def __init__(self, kernel, shapein, axes, fftw_flag, nthreads, **keywords):
        dtype = kernel.dtype
        _FFTWConvolutionOperator.__init__(self, kernel, shapein, dtype, axes,
                                          fftw_flag, nthreads, **keywords)
        n = product(shapein)
        fft = _FFTWComplexForwardOperator(self.shapein, self.axes,
                  self.fftw_flag, self.nthreads, self.dtype, **keywords)
        kernel_fft = _get_kernel_fft(kernel, self.shapein, self.dtype,
                  self.shapein, self.dtype, fft.oplan)
        kernel_fft /= n
        self.__class__ = CompositionOperator
        self.__init__([n, fft.H, DiagonalOperator(kernel_fft), fft])


@real
class _FFTWRealConvolutionOperator(_FFTWConvolutionOperator):
    """
    Convolution by a real kernel.
    
    """
    def __init__(self, kernel, shapein, axes, fftw_flag, nthreads, **keywords):
        dtype = kernel.dtype
        _FFTWConvolutionOperator.__init__(self, kernel, shapein, dtype, axes,
                                          fftw_flag, nthreads, **keywords)
        self.set_rule('.{_FFTWRealConvolutionOperator}', self.
                      _rule_real_convolution, CompositionOperator, globals())
        self.set_rule('.{_FFTWComplexBackwardOperator}', self.
                      _rule_complex_backward, CompositionOperator, globals())
        self.set_rule('{_FFTWComplexForwardOperator}.', self.
                      _rule_complex_forward, CompositionOperator, globals())

        dtype_ = np.dtype('complex' + str(int(dtype.name[5:]) * 2))
        shape_ = self._reshape_to_halfstorage(shapein)
        _load_wisdom()
        with _pool.get(shapein, dtype) as in_:
            with _pool.get(shape_, dtype_) as out:
                t0 = time.time()
                self._fplan = pyfftw.FFTW(in_, out, axes=self.axes,
                                          flags=[self.fftw_flag],
                                          direction='FFTW_FORWARD',
                                          threads=self.nthreads)
                self._bplan = pyfftw.FFTW(out, in_, axes=self.axes,
                                          flags=[self.fftw_flag],
                                          direction='FFTW_BACKWARD',
                                          threads=self.nthreads)

        if time.time() - t0 > FFTW_WISDOM_MIN_DELAY:
            _save_wisdom()

        kernel_fft = _get_kernel_fft(kernel, shapein, dtype, shape_, dtype_,
                                     self._fplan)
        kernel_fft /= product(shapein)
        self.kernel = kernel_fft

    def direct(self, input, output):
        with _pool.get(self.kernel.shape, self.kernel.dtype) as buf:
            self._fplan.update_arrays(input, buf)
            self._fplan.execute()
            buf *= self.kernel
            self._bplan.update_arrays(buf, output)
            self._bplan.execute()

    def transpose(self, input, output):
        with _pool.get(self.kernel.shape, self.kernel.dtype) as buf:
            self._fplan.update_arrays(input, buf)
            self._fplan.execute()
            multiply_conjugate(buf, self.kernel, buf)
            self._bplan.update_arrays(buf, output)
            self._bplan.execute()

    @staticmethod
    def _rule_real_convolution(self, other):
        result = self.copy()
        result.kernel = self.kernel * other.kernel * product(self.shapein)
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
        y = empty(self.shapein, self.dtype)
        self._bplan.update_arrays(self.kernel.copy(), y)
        self._bplan.execute()
        return y

    def _reshape_to_halfstorage(self, shape):
        shape = list(shape)
        shape[self.axes[-1]] = shape[self.axes[-1]] // 2 + 1
        return shape


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
        if self.isalias(input, output):
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
        self.set_rule('.H', lambda s: 
                      HomothetyOperator(1 / product(s.shapein)) *
                      ReverseOperatorFactory(_FFTWComplexBackwardOperator, s,
                                             forward=s))
        self.set_rule('{_FFTWComplexBackwardOperator}.', lambda o,s:
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

        _FFTWComplexOperator.__init__(self, shapein, forward.axes,
                                      forward.fftw_flag,
                                      forward.nthreads, dtype, **keywords)
        self.set_rule('.H', lambda s: 
                      HomothetyOperator(product(s.shapein)) * forward)
        self.set_rule('{_FFTWComplexForwardOperator}.', lambda o,s:
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
        ker_slice = [slice(0,s) for s in kernel.shape]
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
            with open(filename) as f:
                wisdom = f.read()
        except IOError:
            wisdom = ''
        return wisdom
    
    wisdom = [load(f) for f in FFTW_WISDOM_FILES]
    pyfftw.import_wisdom(wisdom)
    _is_fftw_wisdom_loaded = True

def _save_wisdom():
    """ Save wisdom as 3 files. """
    wisdom = pyfftw.export_wisdom()
    for filename, w in zip(FFTW_WISDOM_FILES, wisdom):
        print 
        try:
            os.remove(filename)
        except OSError:
            pass
        if len(w) == 0:
            continue
        with open(filename, 'w') as f:
            f.write(w)

# make FFTW the default
ConvolutionOperator = _FFTWConvolutionOperator
FFTOperator = _FFTWComplexForwardOperator
