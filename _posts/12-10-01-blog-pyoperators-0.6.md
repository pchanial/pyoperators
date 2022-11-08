---
layout: post
title: PyOperators 0.6
excerpt: Release of PyOperators 0.6
category: blog
---

What’s new ?

-   New operators: FFTOperator (for complex transforms) and
    ConvolutionOperator (real and complex kernels) using the
    [pyFFTW](http://hgomersall.github.com/pyFFTW) wrapper, which can
    also seamlessly use the MKL fft routines.
-   New operators: DiagonalNumexprOperator and
    DiagonalNumexprNonSeparableOperator.
-   New IterativeAlgorithm class.
-   New StopCondition class.
-   New OpenMP/MPI preconditioned conjugate gradient solver
-   New utilities: benchmark (to create timing tables), memory\_usage.
-   New context managers: to make block of code uninterruptible or
    interruptible.
-   New ufuncs: multiply\_conjugate, masking (speeding up MaskOperator).
-   New operator decorators to enforce alignment or contiguity.
-   New helper functions: cast, ifind, least\_greater\_multiple,
    renumerate.
-   New functions empty, ones, zeros in the memory manager.
-   Given operators A and B: A / B now translates into A \* B.I. Useful
    when B is a scalar.
-   Automatic download of dependencies using pip.

Under the hood:

-   New memory manager handling alignment and contiguity, rewrite of the
    chaining of operations.
-   Propagate some flags into the associated operators.
-   Propagation of square flags into Addition/MultiplicationOperator.
-   Make BroadcastingOperator more flexible by encapsulating its data
    attribute with get\_data, which enables subclasses not to modify its
    input data during initialisation, but by doing it on-the-fly, hence
    sparing the storage of the modified data.
-   Make Pylab an optional dependency.
-   Ensure that A\*A is A (and not simply equal to), if A is idempotent.
-   Enable inplace operations in NumexprOperator for numexpr version \>=
    2.0.2.
-   TridiagonalOperator cleanup.
-   API changes: decorator universal~~<span
    style="text-align:right;">separable,
    memory.allocate</span>~~\>memory.empty, same\_data-\>isalias.
