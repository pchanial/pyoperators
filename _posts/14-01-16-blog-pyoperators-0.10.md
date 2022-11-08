---
layout: post
title: PyOperators 0.10
excerpt: Release of PyOperators 0.10
category: blog
---

What’s new ?

-   SparseOperator. The sparse storage can be anyone from the
    scipy.sparse package (except the LIL format, which is not suited for
    matrix-vector multiplication)
-   Change binary rule priorities to favour specialized rules and
    subclasses
-   Add mechanism to prevent CompositeOperator from morphing into their
    unique component. This mechanism allow GroupOperator to only have
    one operand
-   Remove DiagonalNumexprNonSeparableOperator, it’s not possible to
    avoid calling get\_data in `__init__` and it complicates the
    broadcasting operators too much
-   API change: the mask convention for PackOperator and UnpackOperator
    is changed. True means kept (similar to Fortran’s pack & unpack).
    Make PackOperator and UnpackOperator subclass BroadcastingBase

Under the hood:

-   Handle infinity in assert\_same
-   Improve rule’s `__str__` when the predicate in a lambda function
-   Add ‘\_reset’ method for Operators
-   Add ‘broadcast’ keyword to strshape
-   Add debugging for the composition and commutative rules
-   Cleanup broadcasting operators, including shape and dtype. Improved
    testing
-   Fix ‘square’ flag when morphing from a DiagonalOperator or a
    MaskOperator to ZeroOperator
-   In ‘todense’, use dtype=int when the operator’s dtype is None
-   Hack to add shapein and shapeout to Operator’s `__repr__`
-   In CommutativeCompositeOperator, get info from the operands before
    applying the simplification rules
-   Add ZeroOperator rule for MultiplicationOperator. Make sure a copy
    of the other operator is done for AdditionOperator
-   In ‘uninterruptible’ contextmanager, ensure that the SIGINT handler
    is always put back in
