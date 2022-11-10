import numpy as np
import pytest
import scipy.signal
from numpy.testing import assert_allclose

from pyoperators import CompositionOperator, ConvolutionOperator, HomothetyOperator
from pyoperators.fft import _FFTWRealConvolutionOperator
from pyoperators.utils.testing import assert_same

from .common import get_associated_array, get_associated_operator

pytest.importorskip('pyfftw')


def assert_convolution(image, kernel):
    ref = scipy.signal.fftconvolve(image, kernel, mode='same')
    convol = ConvolutionOperator(kernel, image.shape)
    con = convol(image)
    assert_allclose(ref, con, atol=1.0e-15)
    assert_allclose(convol.todense().T.conj(), convol.H.todense(), atol=1.0e-15)


def test_convolution_real1():
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
    assert_convolution(image, kernel)


def test_convolution_real2():
    image = np.array([0, 1, 0, 0, 0, 0, 0])
    kernel = [1, 1, 0.5]
    assert_convolution(image, kernel)


@pytest.mark.parametrize('kx', range(1, 4, 2))
def test_convolution_real_1d(kx):
    kshape = (kx,)
    kernel = np.ones(kshape)
    kernel.flat[-1] = 0.5
    for ix in range(kx * 2, kx * 2 + 3):
        ishape = (ix,)
        image = np.zeros(ishape)
        image.flat[image.size // 2] = 1.0
        assert_convolution(image, kernel)


@pytest.mark.parametrize('kx', range(1, 4, 2))
@pytest.mark.parametrize('ky', range(1, 4, 2))
def test_convolution_real_2d(kx, ky):
    kshape = (kx, ky)
    kernel = np.ones(kshape)
    kernel.flat[-1] = 0.5
    for ix in range(kx * 2 + 1, kx * 2 + 3):
        for iy in range(ky * 2 + 1, ky * 2 + 3):
            ishape = (ix, iy)
            image = np.zeros(ishape)
            image[tuple(s // 2 for s in image.shape)] = 1.0
            assert_convolution(image, kernel)


@pytest.mark.parametrize('kx', range(1, 4, 2))
@pytest.mark.parametrize('ky', range(1, 4, 2))
@pytest.mark.parametrize('kz', range(1, 4, 2))
def test_convolution_real_3d(kx, ky, kz):
    kshape = (kx, ky, kz)
    kernel = np.ones(kshape)
    kernel.flat[-1] = 0.5
    for ix in range(kx * 2 + 1, kx * 2 + 3):
        for iy in range(ky * 2 + 1, ky * 2 + 3):
            for iz in range(kz * 2 + 1, kz * 2 + 3):
                ishape = (ix, iy, iz)
                image = np.zeros(ishape)
                image[tuple(s // 2 for s in image.shape)] = 1.0
                assert_convolution(image, kernel)


@pytest.mark.parametrize('ndim', range(1, 5))
def test_convolution_complex(ndim):
    kernel = np.ones(ndim * (3,), complex)
    kernel.flat[-1] = 0.5
    image = np.zeros(ndim * (6,))
    image[tuple(s // 2 for s in image.shape)] = 1.0
    assert_convolution(image, kernel)


@pytest.fixture(scope='module')
def setup_convolution_rules_cmp():
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
    return image, kernel1, kernel2, ref


@pytest.mark.parametrize('kernel1_kind', ['', 'real'])
@pytest.mark.parametrize('kernel2_kind', ['', 'real'])
@pytest.mark.parametrize('swap', [False, True])
def test_convolution_rules_cmp(
    setup_convolution_rules_cmp, kernel1_kind, kernel2_kind, swap
):
    image, kernel1, kernel2, ref = setup_convolution_rules_cmp
    kernel1 = get_associated_array(kernel1, kernel1_kind)
    kernel2 = get_associated_array(kernel2, kernel2_kind)
    if swap:
        kernel1, kernel2 = kernel2, kernel1

    c1 = ConvolutionOperator(kernel1, image.shape)
    c2 = ConvolutionOperator(kernel2, image.shape)
    c = c1 @ c2
    if kernel1.dtype.kind == 'f' and kernel2.dtype.kind == 'f':
        assert isinstance(c, _FFTWRealConvolutionOperator)
    else:
        assert isinstance(c, CompositionOperator)
        assert len(c.operands) == 3
    assert_allclose(c(image.real), ref, atol=1e-08)


@pytest.mark.parametrize('c1_attr', ['', 'T'])
@pytest.mark.parametrize('c2_attr', ['', 'T'])
def test_convolution_rules_add(c1_attr, c2_attr):
    shape = (5, 5)
    kernel1 = np.ones((3, 3))
    kernel2 = np.ones((2, 2))
    c1 = get_associated_operator(ConvolutionOperator(kernel1, shape), c1_attr)
    c2 = get_associated_operator(ConvolutionOperator(kernel2, shape), c2_attr)
    c = c1 + c2
    assert isinstance(c, _FFTWRealConvolutionOperator)
    assert_same(c1.todense() + c2.todense(), c.todense(), atol=5)


@pytest.fixture(scope='module')
def setup_convolution_rules_homothety():
    h = HomothetyOperator(2)
    c = ConvolutionOperator(np.ones((3, 3)), (5, 5))
    ref = c.todense() * h.data
    return h, c, ref


@pytest.mark.parametrize('op_attr', ['', 'T'])
@pytest.mark.parametrize('swap', [False, True])
def test_convolution_rules_homothety(setup_convolution_rules_homothety, op_attr, swap):
    h, c, ref = setup_convolution_rules_homothety
    c = get_associated_operator(c, op_attr)
    ref = get_associated_operator(ref, op_attr)
    op = c @ h if swap else h @ c
    assert_same(op.todense(), ref, atol=5)
