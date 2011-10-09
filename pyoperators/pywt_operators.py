""" Wrap PyWavelets wavelet transforms into Operators.

For now only 1D and 2D wavelets are available.

"""
import numpy as np
import pywt
from .decorators import linear, real
from .core import Operator

# dict of corresponding wavelets
rwavelist = {}
for l in pywt.wavelist():
    if 'bior' in l:
        rwavelist[l] = 'rbio' + l[-3:]
    elif 'rbio' in l:
        rwavelist[l] = 'bior' + l[-3:]
    else:
        rwavelist[l] = l

# Operators factories :

@linear
@real
class Wavelet(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None, **kwargs):
        """
        1D wavelet decomposition and reconstruction. Wavelet coefficients
        are stored in a vector (ndarray with ndim=1).

        Exemples
        --------
        >>> W = Wavelet("haar", level=1, shapein=2)
        >>> W.todense()

        array([[ 0.70710678,  0.70710678],
               [ 0.70710678, -0.70710678]])

        See Also
        --------
        See operators.pywt.MODES docstring for available modes.
        See operators.pywt.wavelist() for available wavelets.
        See operators.pywt.wavedec for the operation performed on input arrays.

        Notes
        -----
        Wrapping around PyWavelet
        """
        self.wavelet = wavelet
        self.rwavelet = rwavelist[self.wavelet]
        self.mode = mode
        self.level = level
        # needed to get sizes of all coefficients
        a = np.zeros(shapein)
        b = pywt.wavedec(a, wavelet, mode=mode, level=level)
        self.sizes = [bi.size for bi in b]
        self.cumsizes = np.zeros(len(self.sizes) + 1)
        np.cumsum(self.sizes, out=self.cumsizes[1:])
        del a, b
        shapeout = sum(self.sizes)
        def direct(x, out):
            coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=self.level)
            out[:] = self.coeffs2vect(coeffs)
        def transpose(x, out):
            coeffs = self.vect2coeffs(x)
            out[:] = pywt.waverec(coeffs, self.rwavelet, mode=self.mode)[:self.shapein[0]]
        super(Wavelet, self).__init__(direct=direct, transpose=transpose,
                                      shapein=shapein, shapeout=shapeout, **kwargs)

    def coeffs2vect(self, coeffs):
        return np.concatenate(coeffs)

    def vect2coeffs(self, vect):
        return [vect[self.cumsizes[i]:self.cumsizes[i + 1]]
                for i in xrange(len(self.sizes))]

@linear
@real
class Wavelet2(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None, **kwargs):
        """
        2D wavelet decomposition and reconstruction. Wavelet coefficients
        are stored in a vector (ndarray with ndim=1).

        Exemple
        -------
        >>> W = Wavelet2("haar", level=1, shapein=(2, 2))
        >>> W.todense()

        array([[ 0.5,  0.5,  0.5,  0.5],
               [ 0.5,  0.5, -0.5, -0.5],
               [ 0.5, -0.5,  0.5, -0.5],
               [ 0.5, -0.5, -0.5,  0.5]])

        See Also
        --------
        See operators.pywt.MODES docstring for available modes.
        See operators.pywt.wavelist() for available wavelets.
        See operators.pywt.wavedec for the operation performed on input arrays.

        Notes
        -----
        Wrapping around PyWavelet
        """
        self.wavelet = wavelet
        self.rwavelet = rwavelist[self.wavelet]
        self.mode = mode
        self.level = level
        # compute shapes and sizes
        a = np.zeros(shapein)
        coeffs = pywt.wavedec2(a, wavelet, mode=mode, level=level)
        approx = coeffs[0]
        details = coeffs[1:]
        self.shapes = [approx.shape,]
        self.shapes +=  [d[i].shape for d in details for i in xrange(3)]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.cumsizes = np.zeros(len(self.sizes) + 1)
        np.cumsum(self.sizes, out=self.cumsizes[1:])
        del approx, details
        shapeout = sum(self.sizes)

        def direct(x, out):
            coeffs = pywt.wavedec2(x, self.wavelet, mode=self.mode, level=self.level)
            out[:] = self.coeffs2vect(coeffs)

        def transpose(x, out):
            coeffs = self.vect2coeffs(x)
            out[:] = pywt.waverec2(coeffs, self.rwavelet, mode=self.mode)[:shapein[0], :shapein[1]]

        super(Wavelet2, self).__init__(direct=direct, transpose=transpose,
                                       shapein=shapein, shapeout=shapeout, **kwargs)

    def coeffs2vect(self, coeffs):
        # distinguish between approximation and details
        approx = coeffs[0]
        details = coeffs[1:]
        # transform 2d arrays into vectors
        vect_coeffs = [approx.ravel(),]
        vect_coeffs += [d[i].ravel() for d in details for i in xrange(3)]
        # put everything into a single coefficient
        return np.concatenate(vect_coeffs)

    def vect2coeffs(self, vect):
        approx = [vect[:self.sizes[0]].reshape(self.shapes[0]),]
        details = [[vect[self.cumsizes[i + j]:self.cumsizes[i + j + 1]].reshape(self.shapes[i + j]) for j in xrange(3)] for i in xrange(1, len(self.sizes), 3)]
        return approx + details
