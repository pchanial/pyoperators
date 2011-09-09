import nose
import numpy as np

from operators import Operator
from operators.utils import isscalar


def assert_is_scalar(o):
    assert isscalar(o)


def assert_is_not_scalar(o):
    assert not isscalar(o)


def test_is_scalar():
    for o in (object, True, 1, 1.0, np.array(1), np.int8, slice, Operator()):
        yield assert_is_scalar, o


def test_is_not_scalar():
    for o in ([], (), np.ones(1), np.ones(2)):
        yield assert_is_not_scalar, o


if __name__ == "__main__":
    nose.run(argv=['', __file__])
