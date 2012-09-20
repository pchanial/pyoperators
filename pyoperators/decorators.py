"""
Define decorators for Operator subclasses. These decorators update
their 'flags' attribute to specify properties such as linear, square etc.
"""

from .memory import MEMORY_ALIGNMENT

def flags(cls, *arg, **keywords):
    """
    Decorator to set any flag.
    """
    base = cls.__mro__[-2]
    base.__dict__['_set_flags'](cls, *arg, **keywords)
    return cls

def linear(cls):
    """
    Decorator for linear operators.
    It sets the 'linear' flags.
    """
    return flags(cls, 'linear')

def square(cls):
    """
    Decorator for square operators, i.e. operators whose input and output
    shapes are identical.
    It sets the 'square' flags.
    """
    return flags(cls, 'square')

def real(cls):
    """
    Decorator for real operators, i.e. operators that are equal to
    their conjugate.
    It sets the 'real' flags.
    """
    return flags(cls, 'real')

def symmetric(cls):
    """
    Decorator for symmetric operators, i.e. operators that are equal to their
    transpose.
    It sets the 'linear', 'square' and 'symmetric' flags.
    """
    return flags(cls, 'symmetric')

def hermitian(cls):
    """
    Decorator for hermitian operators, i.e. operators that are equal to their
    adjoint.
    It sets the 'linear', 'square' and 'hermitian' flags.
    """
    return flags(cls, 'hermitian')

def idempotent(cls):
    """
    Decorator for idempotent operators, i.e. operators whose composition
    by themselves is equal to themselves.
    It sets the 'idempotent' flags.
    """
    return flags(cls, 'idempotent')

def involutary(cls):
    """
    Decorator for involutary operators, i.e. operators whose composition
    by themselves is equal to the identity.
    It sets the 'square' and 'involutary' flags.
    """
    return flags(cls, 'involutary')

def orthogonal(cls):
    """
    Decorator for orthogonal operators, i.e. real operators whose composition
    by their transpose is equal to the identity.
    It sets the 'real', 'linear', 'square' and 'orthogonal' flags.
    """
    return flags(cls, 'orthogonal')

def unitary(cls):
    """
    Decorator for orthogonal operators, i.e. operators whose composition
    by their adjoint is equal to the identity.
    It sets the 'linear', 'square' and 'unitary' flags.
    """
    return flags(cls, 'unitary')

def universal(cls):
    """
    Obsolete decorator, use 'separable' instead.
    """
    return flags(cls, 'separable')

def separable(cls):
    """
    Decorator for separable operators, i.e. operators P which satisfy for  any
    block operator B = [B1, ..., Bn] the property:
        P(B) = [P(B1), ..., P(Bn)] and
        B(P) = [B1(P), ..., Bn(P)]
    It sets the 'separable' flags.
    """
    return flags(cls, 'separable')

def inplace(cls):
    """
    Decorator for inplace operators, i.e operators that can handle input and
    output pointing to the same memory location (though the input and output
    size may be different).
    It sets the 'inplace' attribute to True.
    """
    return flags(cls, 'inplace')

def aligned(cls):
    """
    Decorator to ensure that both input and output of the operator are
    aligned in memory. It sets the alignment_input and alignment_output
    attributes to True.
    """
    return flags(cls, {'alignment_input':MEMORY_ALIGNMENT,
                       'alignment_output':MEMORY_ALIGNMENT})

def aligned_input(cls):
    """
    Decorator to ensure that operator's input is aligned in memory.
    It sets the alignment_input attribute to True.
    """
    return flags(cls, {'alignment_input':MEMORY_ALIGNMENT})

def aligned_output(cls):
    """
    Decorator to ensure that operator's output is aligned in memory.
    It sets the alignment_output attribute to True.
    """
    return flags(cls, {'alignment_output':MEMORY_ALIGNMENT})

def contiguous(cls):
    """
    Decorator to ensure that both input and output of the operator are
    C-contiguous in memory. It sets the contiguous_input and contiguous_output
    attributes to True.
    """
    return flags(cls, 'contiguous_input,contiguous_output')

def contiguous_input(cls):
    """
    Decorator to ensure that operator's input is C-contiguous in memory.
    It sets the contiguous_input attribute to True.
    """
    return flags(cls, 'contiguous_input')

def contiguous_output(cls):
    """
    Decorator to ensure that operator's output is C-contiguous in memory.
    It sets the contiguous_output attribute to True.
    """
    return flags(cls, 'contiguous_output')
