=========
Operators
=========

The operators package defines Operators which are functions with a
shape and dtype, and linear Operators which behave likes matrices
but with close to no storage footprints.

Getting started
===============

To define an Operator one needs to define a direct functions
which will replace the usual matrix vector operation :

>>> def f(x, out):
...     out[:] = 2 *x
...

Then, you can instantiate an Operator:

>>> A = operators.Operator(direct=f, flags={"LINEAR":True, "SYMMETRIC":True})

This operator do not have a shape :

>>> A(ones(5))
Info: Allocating (5,) float64 = 40 bytes in Operator.
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(ones(2))
Info: Allocating (2,) float64 = 16 bytes in Operator.
array([ 2.,  2.])

To output a corresponding dense matrix, one need a linear operator and a shape:

>>> A.todense(shapein=2,)
array([[ 2.,  0.],
[ 0.,  2.]])

Operators do not have to be linear. If they are not they can't be seen
as matrices.  Some operators are already predefined, such as the
IdentityOperator, the DiagonalOperator or the nonlinear
ClippingOperator.

The previous A matrix could be defined more easily like this :

>>> A = 2 * operators.I

where I is the identity Operator with no shape.

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

- numpy >= 1.3
- scipy >= 0.9

Optional requirements:

- PyWavelets : wavelet transforms
