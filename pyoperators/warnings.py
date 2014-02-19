from __future__ import absolute_import
import warnings
from warnings import warn


class PyOperatorsWarning(UserWarning):
    pass


class PyOperatorsDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter('always', category=PyOperatorsWarning)
warnings.simplefilter('module', category=PyOperatorsDeprecationWarning)
del warnings
