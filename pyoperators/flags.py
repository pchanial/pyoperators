"""
Define the Flags class and the decorators for Operator subclasses.
These decorators update their 'flags' attribute to specify properties such as
linear, square etc.

"""
from collections import namedtuple


class Flags(namedtuple(
        'Flags',
        ['linear',
         'square',      # shapein == shapeout
         'real',        # o.C = o
         'symmetric',   # o.T = o
         'hermitian',   # o.H = o
         'idempotent',  # o * o = o
         'involutary',  # o * o = I
         'orthogonal',  # o * o.T = I
         'unitary',     # o * o.H = I
         'separable',   # o*[B1...Bn] = [o*B1...o*Bn]
         'aligned_input',      # aligned input requirement
         'aligned_output',     # aligned output requirement
         'contiguous_input',   # contiguous input requirement
         'contiguous_output',  # contiguous output requirement
         'inplace',            # handle in-place operation
         'outplace',           # handle out-of-place operation
         'update_output',      # handle operations on output
         'destroy_input',      # input modification in out-of-place operation
         'shape_input',
         'shape_output'])):
    """ Informative flags about the operator. """
    def __new__(cls):
        t = 15*(False,) + (True, False, False, '', '')
        return super(Flags, cls).__new__(cls, *t)

    def __str__(self):
        n = max(len(f) for f in self._fields)
        fields = ['  ' + f.upper().ljust(n) + ' : ' for f in self._fields]
        return '\n'.join([f + str(v) for f, v in zip(fields, self)])

    def __repr__(self):
        n = max(len(f) for f in self._fields)
        fields = [f.ljust(n) + '= ' for f in self._fields]
        return type(self).__name__ + '(\n  ' + ',\n  '.join(
            f + repr(v) for f, v in zip(fields, self)) + ')'


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
    It sets the 'linear' and 'symmetric' flags. Note that implicit shape
    symmetric operators do not have to be square.

    """
    return flags(cls, 'symmetric')


def hermitian(cls):
    """
    Decorator for hermitian operators, i.e. operators that are equal to their
    adjoint.
    It sets the 'linear' and 'hermitian' flags. Note that implicit shape
    hermitian operators do not have to be square.

    """
    return flags(cls, 'hermitian')


def idempotent(cls):
    """
    Decorator for idempotent operators, i.e. operators whose composition
    by themselves is equal to themselves.
    It sets the 'idempotent' flag.

    """
    return flags(cls, 'idempotent')


def involutary(cls):
    """
    Decorator for involutary operators, i.e. operators whose composition
    by themselves is equal to the identity.
    It sets the 'involutary' flag. Note that implicit shape involutary
    operators do not have to be square.

    """
    return flags(cls, 'involutary')


def orthogonal(cls):
    """
    Decorator for orthogonal operators, i.e. real operators whose composition
    by their transpose is equal to the identity.
    It sets the 'real', 'linear' and 'orthogonal' flags. Note that implicit
    shape orthogonal operators do not have to be square.

    """
    return flags(cls, 'orthogonal')


def unitary(cls):
    """
    Decorator for orthogonal operators, i.e. operators whose composition
    by their adjoint is equal to the identity.
    It sets the 'linear' and 'unitary' flags. Note that implicit shape
    unitary operators do not have to be square.

    """
    return flags(cls, 'unitary')


def aligned(cls):
    """
    Decorator to ensure that both input and output of the operator are
    aligned in memory. It sets the 'alignment_input' and 'alignment_output'
    attributes to True.

    """
    return flags(cls, 'aligned_input,aligned_output')


def aligned_input(cls):
    """
    Decorator to ensure that operator's input is aligned in memory.
    It sets the 'alignment_input' attribute to True.

    """
    return flags(cls, 'aligned_input')


def aligned_output(cls):
    """
    Decorator to ensure that operator's output is aligned in memory.
    It sets the 'alignment_output' attribute to True.

    """
    return flags(cls, 'aligned_output')


def contiguous(cls):
    """
    Decorator to ensure that both input and output of the operator are
    C-contiguous in memory. It sets the 'contiguous_input' and
    'contiguous_output' attributes to True.

    """
    return flags(cls, 'contiguous_input,contiguous_output')


def contiguous_input(cls):
    """
    Decorator to ensure that operator's input is C-contiguous in memory.
    It sets the 'contiguous_input' attribute to True.

    """
    return flags(cls, 'contiguous_input')


def contiguous_output(cls):
    """
    Decorator to ensure that operator's output is C-contiguous in memory.
    It sets the 'contiguous_output' attribute to True.

    """
    return flags(cls, 'contiguous_output')


def destroy_input(cls):
    """
    Decorator specifying that during an out-of-place operation, the input
    buffer may be altered. It sets the 'destroy_input' attribute to True

    """
    return flags(cls, 'destroy_input')


def inplace(cls):
    """
    Decorator for inplace operators, i.e operators that can handle input and
    output pointing to the same memory location (though the input and output
    size may be different).
    It sets the 'inplace' attribute to True.

    """
    return flags(cls, 'inplace')


def inplace_only(cls):
    """
    Decorator for inplace operators, i.e operators that can handle input and
    output pointing to the same memory location (though the input and output
    size may be different).
    It sets the 'inplace' attribute to True and the 'outplace' to False.

    """
    return flags(cls, inplace=True, outplace=False)


def separable(cls):
    """
    Decorator for separable operators, i.e. operators P which satisfy for  any
    block operator B = [B1, ..., Bn] the property:
        P(B) = [P(B1), ..., P(Bn)] and
        B(P) = [B1(P), ..., Bn(P)]
    It sets the 'separable' flags.

    """
    return flags(cls, 'separable')


def update_output(cls):
    """
    Decorator for operators that can update the output.
    It sets the 'update_output' flag.

    """
    return flags(cls, 'update_output')
