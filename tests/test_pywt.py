import pytest
from numpy.testing import assert_almost_equal

from pyoperators.operators_pywt import Wavelet2dOperator, WaveletOperator

pywt = pytest.importorskip('pywt')
transforms = pywt.wavelist(kind='discrete')
levels = [2]


@pytest.mark.parametrize('name', transforms)
@pytest.mark.parametrize('level', levels)
@pytest.mark.parametrize('size', [(32,)])
def test_wavelet_transpose(name, level, size):
    W = WaveletOperator(name, level=level, shapein=size)
    assert_almost_equal(W.todense(), W.T.todense().T)


@pytest.mark.parametrize('name', transforms)
@pytest.mark.parametrize('level', levels)
@pytest.mark.parametrize('shape', [(4, 4)])
def test_wavelet2d_transpose(name, level, shape):
    W = Wavelet2dOperator(name, level=level, shapein=shape, mode='periodization')
    assert_almost_equal(W.todense(), W.T.todense().T)
