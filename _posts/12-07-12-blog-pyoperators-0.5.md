---
layout: post
title: PyOperators 0.5
excerpt: Release of PyOperators 0.5
category: blog
---

What’s new ?

-   New operators: ReductionOperator, SumOperator, ProductOperator,
    MinOperator, MaxOperator, MinMaxOperator.
-   New operator IntegrationTrapezeWeightOperator.
-   New composite operator GroupOperator: same as CompositionOperator,
    but without the associativity rules.
-   New C-ufuncs, used in SoftThresholdingOperator and
    HardThresholdingOperator.
-   Improved asoperator: handle callable, ufunc, matrices, scalars, add
    ‘constant’ keyword to tell whether scalars should be translated into
    HomothetyOperator or ConstantOperator.
-   RoundOperator: add methods ‘rhtmi’ (round half to minus infinity),
    ‘rhtpi’ (round half to plus infinity).
-   Make todense method work with uncontrained output shape operators.
-   Set dtype to None for DiagonalOperators with only 1 and 0 on the
    diagonal.
-   Set ‘square’ flag automatically for ufuncs. Infer ‘real’ flag from
    ufunc’s types.
-   Fix flags shape\_input, shape\_output in
    CommutativeCompositeOperator and CompositionOperator.
-   Add ‘idempotent’ flag to RoundOperator and HardThresholdingOperator.

Under the hood:

-   Make composite operators behave like regular Operators (in terms of
    attribute handling and merging) to enable subclassing them.
-   Implement CompositionOperator and CommutativeCompositeOperator
    associativity through binary rules.
-   Add helper routine find in utils.
-   Add skiptest decorator.
-   Speed up all\_eq helper routine.
