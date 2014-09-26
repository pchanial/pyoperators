===========
PyOperators
===========

The PyOperators package defines operators and solvers for high-performance computing. These operators are multi-dimensional functions with optimised and controlled memory management. If linear, they behave like matrices with a sparse storage footprint.

More documentation can be found here: http://pchanial.github.io/pyoperators.

Getting started
===============

To define an operator, one needs to define a direct function
which will replace the usual matrix-vector operation:

>>> def f(x, out):
...     out[...] = 2 * x

Then, you can instantiate an ``Operator``:

>>> A = pyoperators.Operator(direct=f, flags='symmetric')

An alternative way to define an operator is to define a subclass:

>>> from pyoperators import flags, Operator
... @flags.symmetric
... class MyOperator(Operator):
...     def direct(x, out):
...         out[...] = 2 * x
...
... A = MyOperator()

This operator does not have an explicit shape, it can handle inputs of any shape:

>>> A(np.ones(5))
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(np.ones((2,3)))
array([[ 2.,  2.,  2.],
       [ 2.,  2.,  2.]])

By setting the ``symmetric`` flag, we ensure that A's transpose is A:

>>> A.T is A
True

For non-explicit shape operators, we get the corresponding dense matrix by specifying the input shape:

>>> A.todense(shapein=2)
array([[2, 0],
       [0, 2]])

Operators do not have to be linear. Many operators are already `predefined <http://pchanial.github.io/pyoperators/2000/doc-operators/#list>`_, such as the ``IdentityOperator``, the ``DiagonalOperator`` or the nonlinear ``ClipOperator``.

The previous ``A`` matrix could be defined more easily like this:

>>> from pyoperators import I
>>> A = 2 * I

where ``I`` is the identity operator with no explicit shape.

Operators can be combined together by addition, element-wise multiplication or composition. Note that the operator ``*`` stands for matrix multiplication if the two operators are linear, or for element-wise multiplication otherwise:

>>> from pyoperators import I, DiagonalOperator
>>> B = 2 * I + DiagonalOperator(range(3))
>>> B.todense()
array([[2, 0, 0],
       [0, 3, 0],
       [0, 0, 4]])

Algebraic rules can easily be attached to operators. They are used to simplify expressions to speed up their execution. The ``B`` Operator has been reduced to:

>>> B
DiagonalOperator(array([2, ..., 4], dtype=int64), broadcast='disabled', dtype=int64, shapein=3, shapeout=3)

Many simplifications are available. For instance:

>>> from pyoperators import Operator
>>> C = Operator(flags='idempotent,linear')
>>> C * C is C
True
>>> D = Operator(flags='involutary')
>>> D(D)
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
