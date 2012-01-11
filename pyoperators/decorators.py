"""
Define decorators for Operator subclasses. These decorators update
their 'flags' attribute to specify properties such as linear, square etc.
"""

def linear(cls):
    """
    Decorator for linear operators.
    It sets the 'linear' flags.
    """
    cls._set_flags(cls, 'linear')
    return cls

def square(cls):
    """
    Decorator for square operators, i.e. operators whose input and output
    shapes are identical.
    It sets the 'square' flags.
    """
    cls._set_flags(cls, 'square')
    return cls

def real(cls):
    """
    Decorator for real operators, i.e. operators that are equal to
    their conjugate.
    It sets the 'real' flags.
    """
    cls._set_flags(cls, 'real')
    return cls

def symmetric(cls):
    """
    Decorator for symmetric operators, i.e. operators that are equal to their
    transpose.
    It sets the 'linear', 'square' and 'symmetric' flags.
    """
    cls._set_flags(cls, 'symmetric')
    return cls

def hermitian(cls):
    """
    Decorator for hermitian operators, i.e. operators that are equal to their
    adjoint.
    It sets the 'linear', 'square' and 'hermitian' flags.
    """
    cls._set_flags(cls, 'hermitian')
    return cls

def idempotent(cls):
    """
    Decorator for idempotent operators, i.e. operators whose composition
    by themselves is equal to themselves.
    It sets the 'idempotent' flags.
    """
    cls._set_flags(cls, 'idempotent')
    return cls

def involutary(cls):
    """
    Decorator for involutary operators, i.e. operators whose composition
    by themselves is equal to the identity.
    It sets the 'square' and 'involutary' flags.
    """
    cls._set_flags(cls, 'involutary')
    return cls
    
def orthogonal(cls):
    """
    Decorator for orthogonal operators, i.e. real operators whose composition
    by their transpose is equal to the identity.
    It sets the 'real', 'linear', 'square' and 'orthogonal' flags.
    """
    cls._set_flags(cls, 'orthogonal')
    return cls
    
def unitary(cls):
    """
    Decorator for orthogonal operators, i.e. operators whose composition
    by their adjoint is equal to the identity.
    It sets the 'linear', 'square' and 'unitary' flags.
    """
    cls._set_flags(cls, 'unitary')
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
