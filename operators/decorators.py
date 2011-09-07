def linear(cls):
    cls._validate_flags(cls, {'LINEAR': True})
    return cls


def square(cls):
    cls._validate_flags(cls, {'SQUARE': True})
    return cls


def real(cls):
    cls._validate_flags(cls, {'REAL': True})
    return cls


def symmetric(cls):
    cls._validate_flags(cls, {'LINEAR': True, 'SQUARE': True, 'SYMMETRIC': True})
    return cls


def hermitian(cls):
    cls._validate_flags(cls, {'LINEAR': True, 'SQUARE': True, 'HERMITIAN': True})
    return cls


def idempotent(cls):
    cls._validate_flags(cls, {'SQUARE': True, 'IDEMPOTENT': True})
    return cls


def orthogonal(cls):
    cls._validate_flags(cls, {'LINEAR': True, 'SQUARE': True, 'ORTHOGONAL': True})
    return cls


def unitary(cls):
    cls._validate_flags(cls, {'LINEAR': True, 'SQUARE': True, 'UNITARY': True})
    return cls


def involutary(cls):
    cls._validate_flags(cls, {'SQUARE': True, 'INVOLUTARY': True})
    return cls
