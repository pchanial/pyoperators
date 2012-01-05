"""
Define decorators for Operator subclasses. These decorators update
their 'flags' attribute to specify properties such as linear, square etc.
"""

def flags(**keywords):
    """
    Decorator that sets operator's flags.
    """
    def func(cls):
        cls._set_flags(cls, keywords)
        return cls
    return func

def linear(cls):
    """
    Decorator for linear operators.
    It sets the LINEAR flags.
    """
    cls._set_flags(cls, {'LINEAR':True})
    return cls

def square(cls):
    """
    Decorator for square operators, i.e. operators whose input and output
    shapes are identical.
    It sets the SQUARE flags.
    """
    cls._set_flags(cls, {'SQUARE':True})
    return cls

def real(cls):
    """
    Decorator for real operators, i.e. operators that are equal to
    their conjugate.
    It sets the REAL flags.
    """
    cls._set_flags(cls, {'REAL':True})
    return cls

def symmetric(cls):
    """
    Decorator for symmetric operators, i.e. operators that are equal to their
    transpose.
    It sets the LINEAR, SQUARE and SYMMETRIC flags.
    """
    cls._set_flags(cls, {'SYMMETRIC':True})
    return cls

def hermitian(cls):
    """
    Decorator for hermitian operators, i.e. operators that are equal to their
    adjoint.
    It sets the LINEAR, SQUARE and HERMITIAN flags.
    """
    cls._set_flags(cls, {'HERMITIAN':True})
    return cls

def idempotent(cls):
    """
    Decorator for idempotent operators, i.e. operators whose composition
    by themselves is equal to themselves.
    It sets the SQUARE and IDEMPOTENT flags.
    """
    cls._set_flags(cls, {'IDEMPOTENT':True})
    return cls

def involutary(cls):
    """
    Decorator for involutary operators, i.e. operators whose composition
    by themselves is equal to the identity.
    It sets the SQUARE and INVOLUTARY flags.
    """
    cls._set_flags(cls, {'INVOLUTARY':True})
    return cls
    
def orthogonal(cls):
    """
    Decorator for orthogonal operators, i.e. real operators whose composition
    by their transpose is equal to the identity.
    It sets the REAL, LINEAR, SQUARE and ORTHOGONAL flags.
    """
    cls._set_flags(cls, {'ORTHOGONAL':True})
    return cls
    
def unitary(cls):
    """
    Decorator for orthogonal operators, i.e. operators whose composition
    by their adjoint is equal to the identity.
    It sets the LINEAR, SQUARE and UNITARY flags.
    """
    cls._set_flags(cls, {'UNITARY':True})
    return cls
    
def inplace(cls):
    """
    Decorator for inplace operators, i.e operators that can handle input and
    output pointing to the same memory location (though the input and output
    size may be different).
    It sets the 'inplace' attribute to True.
    """
    cls.inplace = True
    return cls
