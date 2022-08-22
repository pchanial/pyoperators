import numpy as np
import pytest

from pyoperators.config import MEMORY_ALIGNMENT
from pyoperators.memory import MemoryPool, empty
from pyoperators.utils import tointtuple

buffers = [
    empty(10),
    empty((5, 2)),
    empty(20)[::2],
    empty(11)[1:],
    empty(21)[1:].reshape((10, 2))[::2, :],
]
alignments = 3 * [True] + [False, False]
contiguities = [_.flags.contiguous for _ in buffers]


def assert_contiguous(x):
    assert x.flags.contiguous


def assert_aligned(x):
    assert address(x) % MEMORY_ALIGNMENT == 0


def address(l):
    if isinstance(l, np.ndarray):
        return l.__array_interface__['data'][0]
    return [address(_) for _ in l]


@pytest.mark.parametrize('shape', [10, (10,), (2, 10), (3, 3, 3)])
@pytest.mark.parametrize('dtype', [float, np.int8, complex])
def test_empty(shape, dtype):
    value = empty(shape, dtype)
    assert value.shape == tointtuple(shape)
    assert value.dtype == dtype
    assert_aligned(value)
    assert_contiguous(value)


@pytest.mark.parametrize('buffer', buffers)
def test_set(buffer):
    pool = MemoryPool()
    a = np.empty(9)
    c = np.empty(11)
    pool.add(a)
    pool.add(c)

    with pool.set(buffer):
        assert address(pool._buffers) == address([a, buffer, c])
    assert address(pool._buffers) == address([a, c])


@pytest.fixture(scope='module')
def pool_test_get():
    pool = MemoryPool()
    pa = empty(9)
    pc = empty(11)
    pool.add(pa)
    pool.add(pc)
    return pool, pa, pc


@pytest.mark.parametrize(
    'buffer, aligned, contiguous', zip(buffers, alignments, contiguities)
)
@pytest.mark.parametrize('req_shape', [(10,), (5, 2), (2, 5)])
@pytest.mark.parametrize('req_aligned', [False, True])
@pytest.mark.parametrize('req_contiguous', [False, True])
def test_get_old(
    pool_test_get, buffer, aligned, contiguous, req_shape, req_aligned, req_contiguous
):
    pool, pa, pc = pool_test_get

    with pool.set(buffer):
        with pool.get(req_shape, float, req_aligned, req_contiguous) as v:
            assert v.shape == req_shape
            if req_aligned:
                assert_aligned(v)
            if req_contiguous:
                assert_contiguous(v)
            if (
                req_aligned > aligned
                or req_contiguous
                and not contiguous
                or not contiguous
                and req_shape != buffer.shape
            ):
                assert address(pool._buffers) == address([pa, buffer])
            else:
                assert address(pool._buffers) == address([pa, pc])

        assert address(pool._buffers) == address([pa, buffer, pc])

    assert address(pool._buffers) == address([pa, pc])


@pytest.fixture(scope='module')
def pool_test_new_entry():
    pool = MemoryPool()
    a = empty(12)
    b = empty(20)
    pool.add(a)
    pool.add(b)
    yield {
        'pool': pool,
        'a': a,
        'b': b,
    }


@pytest.mark.parametrize('i, shape', enumerate([4, 12 - 1, 20 - 1]))
def test_new_entry(mocker, pool_test_new_entry, i, shape):
    mocker.patch('pyoperators.config.MEMORY_TOLERANCE', 1.0)
    pool = pool_test_new_entry['pool']
    a = pool_test_new_entry['a']
    b = pool_test_new_entry['b']

    with pool.get(shape, float):
        pass
    assert len(pool) == 3 + i
