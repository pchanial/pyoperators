---
layout: post
title: 3. Solvers
category: doc
---
{::options parse_block_html="true" /}

A solver is a function that attempts to solve the equation `A x = b`.

3.1. <a name="pcg_solver"></a> Preconditioned Conjugate Gradient Solver
-----------------------------------------------------------------------

TBD

3.2. <a name="scipy_solvers"></a> Scipy solvers
-----------------------------------------------

Operators implement the `scipy.sparse.linalg.LinearOperator` interface,
so the following solvers are useable as-is, as long as the operator has
explicit input and output shapes.

<div class="definition">

| `bicg` | Use BIConjugate Gradient iteration |  
| `bicgstab` | Use BIConjugate Gradient STABilized iteration |  
| `cg` | Use Conjugate Gradient iteration |  
| `cgs` | Use Conjugate Gradient Squared iteration |  
| `gmres` | Use Generalized Minimal RESidual iteration |  
| `lgmres` | Solve a matrix equation using the LGMRES algorithm |  
| `minres` | Use MINimum RESidual iteration |  
| `qmr` | Use Quasi-Minimal Residual iteration |

</div>
<hr>

Since an `Operator` is callable, it can also readily be used by the
minimisers in `scipy.optimize`, although the (soon-to-disappear) lack of
an implementation of norms as an `Operator` makes these solvers not so
useful.

<hr>

Note that none of these solvers handle MPI-distributed unknowns.
