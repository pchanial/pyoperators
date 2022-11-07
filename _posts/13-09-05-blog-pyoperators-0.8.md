---
layout: post
title: PyOperators 0.8
excerpt: Release of PyOperators 0.8
category: blog
---

What’s new ?

-   New operators To1dOperator, ToNdOperator for 1-d to N-d indexing
    conversions.
-   SymmetricBandToeplitzOperator, useful to represent stationary
    (inverse) covariance matrices.
-   Add helpers assert\_is\_type, assert\_same, complex\_dtype\_for,
    izip\_broadcast

Under the hood:

-   Simplify alignment handling
-   API change: Wavelet2Operator -\> Wavelet2dOperator
