---
layout: post
title: PyOperators 0.9
excerpt: Release of PyOperators 0.9
category: blog
---

What’s new ?

-   Rotation2dOperator for rotations in a plane
-   Rotation3dOperator, about 1 axis (3 conventions), 2 axes (6
    conventions, including intrinsic and extrinsic rotations) or 3 axes
    (12 conventions)
-   Spherical2CartesianOperator & Cartesian2SphericalOperator with
    conventions ‘zenith,azimuth’, ‘azimuth,zenith’, ‘elevation,azimuth’
    and ‘azimuth,elevation’
-   NormalizeOperator to obtain unit vectors
-   DegreesOperator & RadiansOperator for degrees/radians conversions
-   Implement `__ne__` for Operators
-   Add helper function float\_dtype. Rename
    complex\_dtype\_for-\>complex\_dtype
-   Add helper functions one, pi and zero, which return a value of the
    specified dtype.
-   API change in Operator’s rule:  
    - in unary rules, use ‘C’, ‘T’, ‘H’ or ‘I’ instead of ‘.C’, ‘.T’,
    ‘.H’ or ‘.I’  
    - binary rule subjects are now specified by a 2-tuple  
    - use Operator subclass type instead of the string ‘{MyOperator}’
-   API change: IntegrationTrapezeWeightOperator -\>
    IntegrationTrapezeOperator

Under the hood:

-   benchmark and memory\_usage functions moved to a distinct
    distribution pybenchmarks
-   Add test and coverage commands in setup.py
-   Automatically compute adjoint from transpose, transpose from
    adjoint, inverse\_adjoint from inverse\_transpose and
    inverse\_tranpose from inverse\_adjoint
-   Fix attribute copy during scalar absorption
-   In binary rules, use ‘==’ instead of ‘is’ in subject matching
-   In assert\_same, allow broadcasting for integer types
-   Remove obsolete Operator’s isalias method
-   Make shape\_input, shape\_output and inplace\_reduction non-settable
    flags
-   Take forward FFTW instance dtype into account for the backward
    instance
-   Remove DirectOperatorFactory and ReverseOperatorFactory
-   DenseOperator output and input shapes can now be implicit. Don’t
    rely on this feature yet, it will be changed in future versions.
