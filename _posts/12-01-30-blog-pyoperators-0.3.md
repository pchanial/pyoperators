---
layout: post
title: PyOperators 0.3
excerpt: Release of PyOperators 0.3
category: blog
---

What’s new ?

- DistributionGlobalOperator: operator for distributing global inputs via MPI
- DistributionIdentityOperator: operator for MPI reductions
- DenseOperator: an operator for dense operations
- allocation messages are now switched off. To print them back: `pyoperators.memory.verbose=True`
- Implement [binary]({{ site.baseurl }}/2000/doc-operators/\#binary\_rules rule priorities
- Make the result of a binary rule compatible with the in/out attributes of the pair of operators
- implement [unary]({{ site.baseurl }}/2000/doc-operators/\#unary\_rules rules instead of the less flexible ‘associated\_operators’ mechanism
- add shape\_input, shape\_output, inplace and inplace\_reductions to operator’s [flags]({{ site.baseurl }}/2000/doc-operators/\#operator\_flags
- DirectOperatorFactory and ReverseOperatorFactory
- ZeroOperator to handle inplace reductions
- API changes: block operators must be instanciated with a `axisin`(`out`) or `new_axisin`(`out`) keyword, the default call without any keyword is reserved for future free partitioning
