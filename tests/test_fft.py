from __future__ import division

import itertools
import numpy as np
import scipy.signal

from pyoperators import CompositionOperator, ConvolutionOperator
from pyoperators.fft import _FFTWRealConvolutionOperator
from pyoperators.utils.testing import assert_eq, assert_is_instance


def test_convolution_real():
    def func(image, kernel):
        ref = scipy.signal.convolve(image, kernel, mode='same')
        convol = ConvolutionOperator(kernel, image.shape)
        con = convol(image)
        assert np.allclose(ref, con, atol=1.0e-15)
        assert np.allclose(convol.todense().T, convol.T.todense(), atol=1.0e-15)

    imashape = (7, 7)
    kershape = (3, 3)
    kerorig = (np.array(kershape) - 1) // 2
    kernel = np.zeros(kershape)
    kernel[kerorig[0] - 1 : kerorig[0] + 2, kerorig[1] - 1 : kerorig[1] + 2] = 0.5**4
    kernel[kerorig[0], kerorig[1]] = 0.5
    kernel[kerorig[0] - 1, kerorig[1] - 1] *= 2
    kernel[kerorig[0] + 1, kerorig[1] + 1] = 0
    image = np.zeros(imashape)
    image[3, 3] = 1.0
    yield func, image, kernel

    image = np.array([0, 1, 0, 0, 0, 0, 0])
    kernel = [1, 1, 0.5]
    yield func, image, kernel

    for kx in range(1, 4, 2):
        kshape = (kx,)
        kernel = np.ones(kshape)
        kernel.flat[-1] = 0.5
        for ix in range(kx * 2, kx * 2 + 3):
            ishape = (ix,)
            image = np.zeros(ishape)
            image.flat[image.size // 2] = 1.0
            yield func, image, kernel

    for kx in range(1, 4, 2):
        for ky in range(1, 4, 2):
            kshape = (kx, ky)
            kernel = np.ones(kshape)
            kernel.flat[-1] = 0.5
            for ix in range(kx * 2 + 1, kx * 2 + 3):
                for iy in range(ky * 2 + 1, ky * 2 + 3):
                    ishape = (ix, iy)
                    image = np.zeros(ishape)
                    image[tuple([s // 2 for s in image.shape])] = 1.0
                    yield func, image, kernel

    for kx in range(1, 4, 2):
        for ky in range(1, 4, 2):
            for kz in range(1, 4, 2):
                kshape = (kx, ky, kz)
                kernel = np.ones(kshape)
                kernel.flat[-1] = 0.5
                for ix in range(kx * 2 + 1, kx * 2 + 3):
                    for iy in range(ky * 2 + 1, ky * 2 + 3):
                        for iz in range(kz * 2 + 1, kz * 2 + 3):
                            ishape = (ix, iy, iz)
                            image = np.zeros(ishape)
                            image[tuple([s // 2 for s in image.shape])] = 1.0
                            yield func, image, kernel


def test_convolution_complex():
    def func(image, kernel):
        ref = scipy.signal.fftconvolve(image, kernel, mode='same')
        convol = ConvolutionOperator(kernel, image.shape)
        con = convol(image)
        assert np.allclose(ref, con, atol=1.0e-15)

        assert np.allclose(
            convol.todense().T.conjugate(), convol.H.todense(), atol=1.0e-15
        )

    for ndims in range(1, 5):
        kernel = np.ones(ndims * (3,), complex)
        kernel.flat[-1] = 0.5
        image = np.zeros(ndims * (6,))
        image[tuple([s // 2 for s in image.shape])] = 1.0
        yield func, image, kernel


def test_convolution_rules_cmp():
    shape = (5, 5)
    kernel1 = np.ones((3, 3), complex)
    kernel1.flat[-1] = 0
    kernel2 = np.ones((3, 3), complex)
    kernel2[0, 0] = 0

    image = np.zeros(shape, complex)
    image[2, 2] = 1
    ref = scipy.signal.fftconvolve(
        scipy.signal.fftconvolve(image, kernel1, mode='same'), kernel2, mode='same'
    )
    ref[abs(ref) < 1e-15] = 0
    ref = ref.real

    def func(k1, k2):
        c1 = ConvolutionOperator(k1, shape)
        c2 = ConvolutionOperator(k2, shape)
        c = c1 * c2
        if k1.dtype.kind == 'f' and k2.dtype.kind == 'f':
            assert_is_instance(c, _FFTWRealConvolutionOperator)
        else:
            assert_is_instance(c, CompositionOperator)
            assert_eq(len(c.operands), 3)
        assert np.allclose(c(image.real), ref)

    for k1, k2 in itertools.product((kernel1.real, kernel1), (kernel2.real, kernel2)):
        for k in ([k1, k2], [k2, k1]):
            yield func, k[0], k[1]


def test_convolution_rules_add():
    shape = (5, 5)
    kernel1 = np.ones((3, 3))
    kernel2 = np.ones((2, 2))
    c1 = ConvolutionOperator(kernel1, shape)
    c2 = ConvolutionOperator(kernel2, shape)

    def func(c1, c2):
        c = c1 + c2
        assert_is_instance(c, _FFTWRealConvolutionOperator)

    yield func, c1, c2
