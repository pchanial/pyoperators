import numpy as np

from pyoperators import DiagonalOperator, HomothetyOperator, Operator, memory


def test_str():
    op = Operator()
    op.delete()
    assert str(op) == 'deleted'
    assert repr(op) == 'DeletedOperator()'


def test_collection_reset():
    counter = memory._gc_nbytes_counter
    op = HomothetyOperator(2)
    op.delete()
    assert memory._gc_nbytes_counter - counter == 8
    memory.garbage_collect()
    assert memory._gc_nbytes_counter == 0


def test_collection(mocker):
    mocker.patch('pyoperators.config.GC_NBYTES_THRESHOLD', 8000)
    memory.garbage_collect()
    counter = 0
    for i in range(10):
        data = np.arange(100)
        counter += data.nbytes
        op = DiagonalOperator(data)
        op.delete()
        if i < 9:
            assert memory._gc_nbytes_counter == counter
    assert memory._gc_nbytes_counter == 0
