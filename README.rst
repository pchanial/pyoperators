================
linear_operators
================

What is linear_operators ?
==========================

linear_operators is a package implementing estimation algorithms for
large scale linear problems. The workflow is as follows :

- generate a linear model using LinearOperator instances as buildling
  blocks,

- define a criterion to minimize using this model (e.g. least-square),

- finally, perform minimization on the criterion using a minimization
  algorithm (e.g. conjugate gradient).

Subpackages implements each part of this workflow. Here is a list of
the subpackages :

- interface : Taken from the scipy.sparse package. It implement the
  LinearOperator class which replaces matrices and do not require to
  store any matrix coefficient. It makes use of matrix-vector
  operations instead.

- operators : A set of LinearOperator subclasses implementing various
  linear operation on vectors and their transpose.

- ndoperators : A set of LinarOperator subclasses implementing
  operations on multidimensional arrays.

- iterative : Contains the Criterion class which allows to define objective
  function as well as minimizers such as the conjugate gradient algorithms
  and wrappers to other minimizing algorithms.

- wrappers : define extra LinearOperator subclasses if optional
  dependencies are available.


Requirements
=============

List of requirements:

- numpy >= 1.3
- scipy >= 0.8

Optional requirements:

- PyWavelets
- fht (fast hadamard transform)
