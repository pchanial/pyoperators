=========
Pyoperators
=========

The pyoperators package defines operators and solvers for high-performance computing. These operators are multi-dimensional functions with optimised and controlled memory management. If linear, they behave like matrices with a sparse storage footprint.

Getting started
===============

To define an operator, one needs to define a direct function
which will replace the usual matrix-vector operation:

>>> def f(x, out):
...     out[...] = 2 * x

Then, you can instantiate an ``Operator``:

>>> A = pyoperators.Operator(direct=f, flags='symmetric')

An alternative way to define an operator is to define a subclass:

>>> from pyoperators import decorators, Operator
... @decorators.symmetric
... class MyOperator(Operator):
...     def direct(x, out):
...         out[...] = 2 * x
...
... A = MyOperator()

This operator does not have an explicit shape, it can handle inputs of any shape:

>>> A(ones(5))
Info: Allocating (5,) float64 = 40 bytes in Operator.
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(ones((2,3)))
Info: Allocating (2,3) float64 = 48 bytes in Operator.
array([[ 2.,  2.,  2.],
       [ 2.,  2.,  2.]])

By setting the 'symmetric' flag, we ensure that A's transpose is A:

>>> A.T is A
True

To output a corresponding dense matrix, one needs to specify the input shape:

>>> A.todense(shapein=2)
array([[ 2.,  0.],
       [ 0.,  2.]])

Operators do not have to be linear, but if they are not, they cannot be seen
as matrices. Some operators are already predefined, such as the
``IdentityOperator``, the ``DiagonalOperator`` or the nonlinear
``ClippingOperator``.

The previous ``A`` matrix could be defined more easily like this :

>>> A = 2 * pyoperators.I

where ``I`` is the identity operator with no explicit shape.

Operators can be combined together by addition, element-wise multiplication or composition (note that the ``*`` sign stands for composition):

>>> B = 2 * pyoperators.I + pyoperators.DiagonalOperator(arange(3))
>>> B.todense()
array([[ 2.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0.,  0.,  4.]])

Algebraic rules are used to simplify an expression involving operators, so to speed up its execution:

>>> B
DiagonalOperator(array([ 2.,  3.,  4.]))
>>> C = pyoperators.Operator(flags='idempotent')
>>> C * C is C
True
>>> D = pyoperators.Operator(flags='involutary')
>>> D * D
IdentityOperator()


Requirements
============

List of requirements:

- python 2.6
- numpy >= 1.6
- scipy >= 0.9

Optional requirements:

- numexpr (>= 2.0 is better)
- PyWavelets : wavelet transforms
