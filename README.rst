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

>>> A = operators.Operator(direct=f, flags={"LINEAR":True, "SYMMETRIC":True}, shapein=2,)
>>> A.todense()
array([[ 2.,  0.],
[ 0.,  2.]])

The input function do not assume any shape so we can use it to define another operator:

>>> A5 = operators.Operator(direct=f, flags={"LINEAR":True, "SYMMETRIC":True}, shapein=5,)
>>> A5 * ones(5)
Info: Allocating (5,) float64 = 40 bytes in Operator.
ndarraywrap([ 2.,  2.,  2.,  2.,  2.])

Operators do not have to be linear. If they are not they can't be seen
as matrices.  Some operators are already predefined, such as the
IdentityOperator, the DiagonalOperator or the nonlinear
ClippingOperator.

Operators can be combined together by addition or by operator
multiplication (composition of functions).

Requirements
=============

List of requirements:

- numpy >= 1.3
- scipy >= 0.8

Optional requirements:

- PyWavelets : wavelet transforms
