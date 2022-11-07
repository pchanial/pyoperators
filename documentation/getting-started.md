---
layout: post
title: 1. Getting started
published: true
category: doc
---

To define an operator, one needs to define a direct function which will
replace the usual matrix-vector operation:

```python
>>> def f(x, out):
...     out[...] = 2 * x
```

Then, you can instantiate an `Operator`:

```python
>>> A = Operator(f, flags='symmetric')
```

This operator does not have an explicit shape, it can handle inputs of
any shape:

```python
>>> A(np.ones(5))
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(np.ones((2,3)))
array([[ 2.,  2.,  2.],
       [ 2.,  2.,  2.]])
```

By setting the ‘symmetric’ flag, we ensure that A’s transpose is A:

```python
>>> A.T is A
True
```

To output a corresponding dense matrix, one needs to specify the input
shape:

```python
>>> A.todense(shapein=2)
array([[ 2.,  0.],
       [ 0.,  2.]])
```

Operators do not have to be linear, but if they are not, they cannot be
seen as matrices. Some operators are already predefined, such as the
linear operators `IdentityOperator` and `DiagonalOperator` or the
nonlinear operator `ClippingOperator`.

The previous `A` matrix could be defined more easily like this :

```python
>>> A = 2 * I
```

where `I` is the identity operator with no explicit shape.

Operators can be combined together by addition, multiplication or
composition (note that the `*` sign stands for composition):

```python
>>> B = 2 * I + DiagonalOperator(np.arange(3))
>>> B.todense()
array([[ 2.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0.,  0.,  4.]])
```

Algebraic rules are used to simplify an expression involving operators,
so to speed up its execution:

```python
>>> B
DiagonalOperator(array([ 2.,  3.,  4.]))
```
