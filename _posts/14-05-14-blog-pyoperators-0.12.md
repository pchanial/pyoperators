---
layout: post
title: PyOperators 0.12
excerpt: Release of PyOperators 0.12
category: blog
---

What’s new ?

-   add a `nbytes` property, indicating the operator storage footprint.
-   `delete` method for operators, to force deletion of an operator,
    alongside its associated operators. If the storage footprint of the
    deleted operators exceeds a certain threshold `GC_NBYTES_THRESHOLD`,
    a garbage collection is triggered.
-   ProxyOperator for on-the-fly computations. In a group of proxy
    operators, only one actual operator is cached. The proxy operators
    inherit all the properties from the actual operators.
-   add a context manager `setting` to change a global variable value.

Under the hood:

-   Renamed operator flag ‘inplace\_reduction’ -\> ‘update\_output’. And
    make it a regular flag, i.e. the ‘update\_output’ is no more
    inferred from the `direct` method signature.
-   allow generators in composite operators.
