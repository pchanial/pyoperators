""" Wrap pywt transform into LinearOperators """
import numpy as np
import pywt
from .operators import linear, real
from .core import Operator
from copy import copy

# Operators factories :


@linear
@real
class Wavelet(Operator):
    def __init__(self, wavelet, mode='zpd', level=None, shapein=None, **kwargs):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        a = np.zeros(shapein)
        b = pywt.wavedec(a, wavelet, mode=mode, level=level)
        c = np.concatenate(b)

        def direct(x, out):
            dec_list = pywt.wavedec(x, self.wavelet, mode=self.mode, level=self.level)
            out[:] = np.concatenate(dec_list)

        def transpose(x, out):
            x_list = []
            count = 0
            for el in b:
                n_el = np.asarray(el).size
                x_list.append(np.array(x[count : count + n_el]))
                count += n_el
            out[:] = pywt.waverec(x_list, self.wavelet, mode=self.mode)[: self.shapein]

        shapeout = c.size
        super(Wavelet, self).__init__(
            direct=direct,
            transpose=transpose,
            shapein=shapein,
            shapeout=shapeout,
            **kwargs,
        )


@linear
@real
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
        a = np.zeros(shapein)
        b = pywt.wavedec2(a, wavelet, mode=mode, level=level)
        shapeout = coefs2array(b).shape

        def direct(x, out):
            coefs = pywt.wavedec2(x, wavelet, mode=mode, level=level)
            out[:] = coefs2array(coefs)

        def transpose(x, out):
            coefs = array2coefs(x, b)
            out[:] = pywt.waverec2(coefs, wavelet, mode=mode)[
                : shapein[0], : shapein[1]
            ]

        super(Wavelet2, self).__init__(
            direct=direct,
            transpose=transpose,
            shapein=shapein,
            shapeout=shapeout,
            **kwargs,
        )


# utility functions (to put wavelet coefficients into arrays and back)


def vectorize_coefficients(coefs):
    out = coefs[0].ravel()
    for scale in coefs[1:]:
        out = np.concatenate(
            [
                out,
            ]
            + [scale[i].ravel() for i in xrange(3)]
        )
    return out


def vectors2coefs(x, b):
    p = 0
    q = b[0].size
    coefs = [
        x[p:q].reshape(b[0].shape),
    ]
    for i in xrange(1, len(b)):
        scale = list()
        for j in xrange(3):
            p = copy(q)
            q = p + b[i][j].size
            scale += [
                x[p:q].reshape(b[i][j].shape),
            ]
        coefs.append(scale)
    return coefs


def coefs2array(coefs):
    out = coefs[0]
    for scale in coefs[1:]:
        if out.shape[0] == scale[0].shape[0] + 1:
            out = out[:-1]
        if out.shape[1] == scale[0].shape[1] + 1:
            out = out[:, :-1]
        out = np.vstack((np.hstack((out, scale[0])), np.hstack((scale[1], scale[2]))))
    return out


def array2coefs(a, l):
    ilim = [0, l[0].shape[0]]
    jlim = [0, l[0].shape[1]]
    coefs = [
        a[ilim[0] : ilim[1], jlim[0] : jlim[1]],
    ]
    for i in xrange(1, len(l)):
        ilim.append(ilim[-1] + l[i][0].shape[0])
        jlim.append(jlim[-1] + l[i][0].shape[1])
        scale = (
            a[0 : ilim[-2], jlim[-2] : jlim[-1]],
            a[ilim[-2] : ilim[-1], 0 : jlim[-2]],
            a[ilim[-2] : ilim[-1], jlim[-2] : jlim[-1]],
        )
        coefs.append(scale)
    return coefs
