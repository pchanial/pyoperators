---
layout: post
title: PyOperators 0.7
excerpt: Release of PyOperators 0.7
category: blog
---

What’s new ?

-   Raising an operator to an integer exponent.
-   New operator DifferenceOperator.
-   New Variable class and X instance.
-   Expose pcg function, instead of PCGAlgorithm
-   In a composition, HomothetyOperator has a distinct and more
    efficient handling. It can now be absorbed by the operands.
-   New composition rules for ReshapeOperator, real ConvolutionOperator,
    BlockSliceOperator, DenseOperator
-   New class for the transpose of the real-kernel convolution
    (subclassing the real-kernel convolution), so that composition and
    addition rules can be applied to both the convolution and its
    transpose.

Under the hood:

-   Fix todense for operators requiring aligned arrays.
-   Use site.USER\_BASE instead of \~/.local to store FFTW wisdom and
    make environment variable PYOPERATORSPATH override it.
-   New lanczos implementation.
-   Make EigendecompositionOperator subclass CompositionOperator
-   API change:
    DistributionGlobalOperator-\>MPIDistributionGlobalOperator
