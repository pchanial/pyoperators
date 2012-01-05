=========
Pyoperators
=========

The pyoperators package defines operators which are functions with a
shape and dtype, and linear operators which behave like matrices
with a sparse storage footprint.

Getting started
===============

To define an ``Operator`` one needs to define a direct function
which will replace the usual matrix-vector operation:

>>> def f(x, out):
...     out[...] = 2 *x
...

Then, you can instantiate an ``Operator``:

>>> A = pyoperators.Operator(direct=f, flags={"LINEAR":True, "SYMMETRIC":True})

This operator does not have an explicit shape, it can handle inputs of any shape:

>>> A(ones(5))
Info: Allocating (5,) float64 = 40 bytes in Operator.
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(ones(2))
Info: Allocating (2,) float64 = 16 bytes in Operator.
array([ 2.,  2.])

To output a corresponding dense matrix, one needs to specify the input shape:

>>> A.todense(shapein=2)
array([[ 2.,  0.],
[ 0.,  2.]])

Operators do not have to be linear, but if they are not, they cannot be seen
as matrices. Some operators are already predefined, such as the
IdentityOperator, the DiagonalOperator or the nonlinear
ClippingOperator.

The previous ``A`` matrix could be defined more easily like this :

>>> A = 2 * operators.I

where ``I`` is the identity operator with no explicit shape.

Operators can be combined together by addition or by operator
multiplication (composition of functions) :

>>> B = 2 * operators.I + operators.DiagonalOperator(arange(3))
>>> B.todense()
Info: Allocating (3,) float64 = 24 bytes in AdditionOperator.
array([[ 2.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0.,  0.,  4.]])

Requirements
============

List of requirements:
- python 2.6
- numpy >= 1.6
- scipy >= 0.9

Optional requirements:
- numexpr (>= 2.0 is better)
- PyWavelets : wavelet transforms
