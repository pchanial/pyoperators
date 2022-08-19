import warnings


class PyOperatorsWarning(UserWarning):
    pass


class PyOperatorsDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter('always', category=PyOperatorsWarning)
warnings.simplefilter('module', category=PyOperatorsDeprecationWarning)
del warnings
