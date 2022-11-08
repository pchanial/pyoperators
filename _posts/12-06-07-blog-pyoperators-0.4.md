---
layout: post
title: PyOperators 0.4
excerpt: Release of PyOperators 0.4
category: blog
---

What’s new ?

-   new operators BlockSliceOperator, HardThresholdingOperator and
    SoftThresholdingOperator
-   add commin and commout as operator’s attribute, to deal with input
    and output communicators. Handling of implicit communicators. Add
    propagation mechanism.
-   add MPI helper functions combine, distribute, distribute\_shapes,
    mprint, filter\_comm, as\_mpi
-   add function product
-   when combined with a block operator, split DiagonalOperators and
    ConstantOperators
-   add operator’s flag ‘universal’, used in rules involving operators
    and block operators
-   infer implicit partition during addition, multiplication or
    composition
-   binary rule for DiagonalOperator and BlockOperator, including
    broadcasting
-   API change: broadcasting method is now ‘leftward’ or ‘rightward’
-   implement Operator’s *eq*
-   automatically set ‘inplace’ flag for ufuncs
-   make BlockRowOperator handle arbitrary reduction operation, optimize
    direct method by using the memory manager
-   automatic morphing of DiagonalOperator into IdentityOperator,
    ZeroOperator, HomothetyOperator or MaskOperator according to
    diagonal values
-   fix involutary flag in DiagonalOperators
-   add simplification fules: op + O -\> op and op x I -\> op
