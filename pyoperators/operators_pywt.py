""" Wrap PyWavelets wavelet transforms into Operators.

For now only 1D and 2D wavelets are available.

"""
from __future__ import absolute_import, division, print_function
from .core import Operator, CompositionOperator
from .flags import linear, real
import numpy as np
try:
    import pywt
    # dict of corresponding wavelets
    rwavelist = {}
    for l in pywt.wavelist():
        if 'bior' in l:
            rwavelist[l] = 'rbio' + l[-3:]
        elif 'rbio' in l:
            rwavelist[l] = 'bior' + l[-3:]
        else:
            rwavelist[l] = l
except ImportError:
    pass

__all__ = ['WaveletOperator', 'Wavelet2dOperator']


# doctest nose fixture
def setup_module(module):
    try:
        import pywt
    except ImportError:
        from nose.plugins.skip import SkipTest
        raise SkipTest()


@real
@linear
class WaveletOperator(Operator):
    skip_doctest = True
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None,
                 dtype=float, **keywords):
        """
        1D wavelet decomposition and reconstruction. Wavelet coefficients
        are stored in a vector (ndarray with ndim=1).

        Exemples
        --------
        >>> W = WaveletOperator("haar", level=1, shapein=2)
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
        Wrapping around PyWavelets

        """
        if not isinstance(wavelet, pywt.Wavelet):
            wavelet = pywt.Wavelet(wavelet)
        self.wavelet = wavelet
        self.rwavelet = rwavelist[wavelet.name]
        self.mode = mode
        self.level = level
        # needed to get sizes of all coefficients
        a = np.zeros(shapein)
        b = pywt.wavedec(a, wavelet, mode=mode, level=level)
        self.sizes = [bi.size for bi in b]
        self.cumsizes = np.zeros(len(self.sizes) + 1)
        np.cumsum(self.sizes, out=self.cumsizes[1:])
        shapeout = sum(self.sizes)
        Operator.__init__(self, shapein=shapein, shapeout=shapeout,
                          dtype=dtype, **keywords)
        if self.wavelet.orthogonal:
            self.set_rule('T,.', '1', CompositionOperator)

    def direct(self, x, out):
        coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode,
                              level=self.level)
        out[:] = self._coeffs2vect(coeffs)

    def transpose(self, x, out):
        coeffs = self._vect2coeffs(x)
        out[:] = pywt.waverec(coeffs, self.rwavelet,
                              mode=self.mode)[:self.shapein[0]]

    def _coeffs2vect(self, coeffs):
        return np.concatenate(coeffs)

    def _vect2coeffs(self, vect):
        return [vect[self.cumsizes[i]:self.cumsizes[i + 1]]
                for i in range(len(self.sizes))]


@real
@linear
class Wavelet2dOperator(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None,
                 dtype=float, **keywords):
        """
        2D wavelet decomposition and reconstruction. Wavelet coefficients
        are stored in a vector (ndarray with ndim=1).

        Exemple
        -------
        >>> W = Wavelet2dOperator("haar", level=1, shapein=(2, 2))
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
        if not isinstance(wavelet, pywt.Wavelet):
            wavelet = pywt.Wavelet(wavelet)
        self.wavelet = wavelet
        self.rwavelet = rwavelist[wavelet.name]
        self.mode = mode
        self.level = level
        # compute shapes and sizes
        a = np.zeros(shapein)
        coeffs = pywt.wavedec2(a, wavelet, mode=mode, level=level)
        approx = coeffs[0]
        details = coeffs[1:]
        self.shapes = [approx.shape]
        self.shapes += [d[i].shape for d in details for i in range(3)]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.cumsizes = np.zeros(len(self.sizes) + 1)
        np.cumsum(self.sizes, out=self.cumsizes[1:])
        shapeout = sum(self.sizes)

        Operator.__init__(self, shapein=shapein, shapeout=shapeout,
                          dtype=dtype, **keywords)
        if self.wavelet.orthogonal:
            self.set_rule('T,.', '1', CompositionOperator)

    def direct(self, x, out):
        coeffs = pywt.wavedec2(x, self.wavelet, mode=self.mode,
                               level=self.level)
        out[:] = self._coeffs2vect(coeffs)

    def transpose(self, x, out):
        coeffs = self._vect2coeffs(x)
        rec = pywt.waverec2(coeffs, self.rwavelet, mode=self.mode)
        out[:] = rec[:self.shapein[0], :self.shapein[1]]

    def _coeffs2vect(self, coeffs):
        # distinguish between approximation and details
        approx = coeffs[0]
        details = coeffs[1:]
        # transform 2d arrays into vectors
        vect_coeffs = [approx.ravel()]
        vect_coeffs += [d[i].ravel() for d in details for i in range(3)]
        # put everything into a single coefficient
        return np.concatenate(vect_coeffs)

    def _vect2coeffs(self, vect):
        cs = self.cumsizes
        approx = [vect[:self.sizes[0]].reshape(self.shapes[0])]
        details = [[vect[cs[i + j]:cs[i + j + 1]].reshape(self.shapes[i + j])
                    for j in range(3)] for i in range(1, len(self.sizes), 3)]
        return approx + details
