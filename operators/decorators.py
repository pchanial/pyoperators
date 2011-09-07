def linear(cls):
    cls.flags = cls.flags._replace(LINEAR=True)
    cls._validate_flags(cls)
    return cls

def square(cls):
    cls.flags = cls.flags._replace(SQUARE=True)
    cls._validate_flags(cls)
    return cls

def real(cls):
    cls.flags = cls.flags._replace(REAL=True)
    cls._validate_flags(cls)
    return cls

def symmetric(cls):
    cls.flags = cls.flags._replace(LINEAR=True,
                                   SQUARE=True,
                                   SYMMETRIC=True)
    cls._validate_flags(cls)
    return cls

def hermitian(cls):
    cls.flags = cls.flags._replace(LINEAR=True,
                                   SQUARE=True,
                                   HERMITIAN=True)
    cls._validate_flags(cls)
    return cls

def idempotent(cls):
    cls.flags = cls.flags._replace(SQUARE=True,
                                   IDEMPOTENT=True)
    cls._validate_flags(cls)
    return cls

def orthogonal(cls):
    cls.flags = cls.flags._replace(LINEAR=True,
                                   SQUARE=True,
                                   ORTHOGONAL=True)
    cls._validate_flags(cls)
    return cls
    
def unitary(cls):
    cls.flags = cls.flags._replace(LINEAR=True,
                                   SQUARE=True,
                                   UNITARY=True)
    cls._validate_flags(cls)
    return cls
    
def involutary(cls):
    cls.flags = cls.flags._replace(SQUARE=True,
                                   INVOLUTARY=True)
    cls._validate_flags(cls)
    return cls
    
    
