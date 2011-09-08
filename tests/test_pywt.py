#!/usr/bin/env python
import nose
from numpy import testing

import pywt
import operators
import operators.pywt_operators

sizes = ((32,),)
shapes = ((4, 4),)
wavelist = pywt.wavelist()
levels = [
    2,
]


def check_wavelet_transpose(w, l, s):
    W = operators.pywt_operators.Wavelet(w, level=l, shapein=s)
    testing.assert_array_almost_equal(W.todense(), W.T.todense().T)


def test_wavelet_transpose():
    for s in sizes:
        for w in wavelist:
            for l in levels:
                yield check_wavelet_transpose, w, l, s


def check_wavelet2_transpose(w, l, s):
    W = operators.pywt_operators.Wavelet2(w, level=l, shapein=s, mode='per')
    testing.assert_array_almost_equal(W.todense(), W.T.todense().T)


def test_wavelet2_transpose():
    for s in shapes:
        for w in wavelist:
            for l in levels:
                yield check_wavelet2_transpose, w, l, s


if __name__ == "__main__":
    nose.run(defaultTest=__file__)
