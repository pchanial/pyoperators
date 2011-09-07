""" Wrap pywt transform into LinearOperators """
import numpy as np
import pywt
from .core import Operator, Real, Linear

# Operators factories :

@Linear
@Real
class Wavelet(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None, **kwargs):
        self.wavelet = wavelet
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
            out[:] = pywt.waverec(coeffs, self.wavelet, mode=self.mode)[:self.shapein[0]]
        super(Wavelet, self).__init__(direct=direct, transpose=transpose,
                                      shapein=shapein, shapeout=shapeout, **kwargs)

    def coeffs2vect(self, coeffs):
        return np.concatenate(coeffs)

    def vect2coeffs(self, vect):
        return [vect[self.cumsizes[i]:self.cumsizes[i + 1]]
                for i in xrange(len(self.sizes))]

@Linear
@Real
class Wavelet2(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None, **kwargs):
        """
        2d wavelet decomposition / reconstruction as a NDOperator.

        Notes
        -----
        Does not work with all parameters depending if wavelet coefficients
        can be concatenated as 2d arrays !
        Otherwise, take a look at wavelet2
        """
        self.wavelet = wavelet
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
            coeffs = pywt.wavedec2(x, wavelet, mode=mode, level=level)
            out[:] = self.coeffs2vect(coeffs)

        def transpose(x, out):
            coeffs = self.vect2coeffs(x)
            out[:] = pywt.waverec2(coeffs, wavelet, mode=mode)[:shapein[0], :shapein[1]]

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
