"""
Defines decorators for Operator subclasses. Those decorators update
the subclass flags to specify properties of the operators (linear,
square, ...).
"""

def linear(cls):
    cls._set_flags(cls, {'LINEAR':True})
    return cls

def square(cls):
    cls._set_flags(cls, {'SQUARE':True})
    return cls

def real(cls):
    cls._set_flags(cls, {'REAL':True})
    return cls

def symmetric(cls):
    cls._set_flags(cls, {'LINEAR':True,
                         'SQUARE':True,
                         'SYMMETRIC':True})
    return cls

def hermitian(cls):
    cls._set_flags(cls, {'LINEAR':True,
                         'SQUARE':True,
                         'HERMITIAN':True})
    return cls

def idempotent(cls):
    cls._set_flags(cls, {'SQUARE':True,
                         'IDEMPOTENT':True})
    return cls

def orthogonal(cls):
    cls._set_flags(cls, {'REAL':True,
                         'LINEAR':True,
                         'SQUARE':True,
                         'ORTHOGONAL':True})
    return cls
    
def unitary(cls):
    cls._set_flags(cls, {'REAL':False,
                         'LINEAR':True,
                         'SQUARE':True,
                         'UNITARY':True})
    return cls
    
def involutary(cls):
    cls._set_flags(cls, {'SQUARE':True,
                         'INVOLUTARY':True})
    return cls
    
    
