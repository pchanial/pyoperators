---
layout: post
title: PyOperators 0.11
excerpt: Release of PyOperators 0.11
category: blog
---

What’s new ?

-   New rule manager: new mechanism to enable or disable simplification
    rules inside a `with` context. The rule trigger `none` is
    implemented and it inhibits all algebraic simplifications if set to
    True.
-   API change: the multiplication sign `*` is now context-dependent:
    composition for linear operators and element-wise multiplication
    otherwise.
-   New operator DenseBlockDiagonalOperator. Extra dimensions of arrays
    are interpreted as diagonal blocks. This operator is
    broadcastable:  
    - against vectors: multiplying a `DenseBlockDiagonalOperator`
    initialised with an array of shape (L, 1, M, N) by a tensor of shape
    (P, N) gives a tensor of shape (L, P, M)  
    - against other `DenseBlockDiagonalOperators`: multiplying two of
    them initialised with arrays of shapes (K, 1, M, N) and (L, N, P)
    gives a DenseBlockDiagonalOperator initialised with an array of
    shape (K, L, M, P)
-   `DenseOperator` is now implicit shape and broadcastable: an
    M&times;N `DenseOperator` multiplied by a tensor of shape (P, Q, N)
    gives a tensor of shape (P, Q, M).
-   Make `asoperator` return an explicit shape `DiagonalOperator`,
    `DenseOperator` or `DenseBlockDiagonalOperator` for arrays of
    dimensions 1, 2 or more if the keyword `constant` is not set to
    True.
-   New operators: `PowerOperator`, `ReciprocalOperator`, `SqrtOperator`
    and `SquareOperator`.
-   Helper functions:  
    - `reshape_broadcast` to reshape an array to a broadcastable shape,
    by using 0-strides along the missing dimensions  
    - `broadcast_shapes` to return the shape of the output obtained by
    broadcasting arrays of given shapes  
    - add `reverse` option for `first`, `first_is_not`, `ifirst` and
    `ifirst_is_not`.
-   Improved `__str__` for non-linear operators.
-   New ufunc `abs2`, which returns the square modulus of a complex
    input.
-   DEPRECATED: use `isscalarlike` instead of `isscalar`.

Under the hood:

-   Add `PyOperatorsWarning` and `PyOperatorsDeprecationWarnings`.
-   Rename the `decorators` module -\> `flags`.
-   Move operator’s rules -\> `rules` module.
-   Add `CopyOperator`.
