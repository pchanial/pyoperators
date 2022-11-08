---
layout: post
title: PyOperators 0.13
excerpt: Release of PyOperators 0.13
category: blog
---

What’s new ?

-   Python 3 support
-   add settingerr context manager to contextually set error behaviour
-   add environment variables `PYOPERATORS_NO_MPI` (to prevent importing
    mpi4py on supercomputers such as NERSC’s edison to avoid crashes on
    login nodes), `PYOPERATORS_GC_NBYTES_THRESHOLD`,
    `PYOPERATORS_MEMORY_ALIGNMENT`, `PYOPERATORS_MEMORY_TOLERANCE` and
    `PYOPERATORS_VERBOSE` (see config.py file)
-   add helpers last, last\_is\_not, ilast and ilast\_is\_not, split
-   improved ProxyOperator: operators in a proxy group may have
    different shapes, and can access the operator’s attributes
-   Return algorithm instance in `pcg` ’s output
-   add pool\_threading context manager
-   add MPI rank in strinfo
-   in iterative algorithms, silence MPI processes with rank\>0
-   add timers to measure the time spent in Operator’s `__call__` method
    and in MPI operations

API changes, deprecations:

-   rename `openmp_num_threads` -\> `omp_num_threads`
-   deprecate `distribute` and `distribute_slice`
-   change `isclassattr` calling sequence

Under the hood:

-   common superclass for DiagonalOperator, DiagonalNumexprOperator and
    MaskOperator
-   add `Cartcomm` communicator and `Comm.Create_cart` method in
    fake\_mpi module
-   morph MPIDistributionIdentityOperator into IdentityOperator is
    communicator size is 1
-   allow implicit shape SparseBase subclasss
-   update for scipy 0.14
-   new decorator deprecated
-   add setup.py’s clean command (contributed by G. Vaillant)
-   automatic cythonization of files (contributed by G. Vaillant)
