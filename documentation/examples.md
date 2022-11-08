---
layout: post
title: 5. Examples
category: doc
---

5.1 <a name="data_fusion"></a>Data fusion
-----------------------------------------

This example treats the case where a high resolution high noise
observation and a low resolution low noise observation are available. We
will see how these two observations can be combined and deconvolved.
This example is not about which deconvolution method is the best, so
we’ll choose the simplest method: Tikhonov regularisation using the
Identity matrix, which minimises the total power.  
We write the acquisition as a block column operator, whose blocks are
the individual methods of observation.

```python
from __future__ import division
import numpy as np
import scipy
from pyoperators import (BlockColumnOperator, BlockDiagonalOperator,
                         ConvolutionOperator, pcg, I)

x0 = scipy.misc.lena().astype(float)
kernel_sizes = [3, 15]
sigmas = [50, 10]

Hs = [ConvolutionOperator(np.ones((k,k)) / k**2, x0.shape) for k in kernel_sizes]
ns = [np.random.standard_normal(x0.shape) * s for s in sigmas]
invNs = [1 / s**2 for s in sigmas]

H = BlockColumnOperator(Hs, new_axisout=0)
n = np.concatenate(ns).reshape((2,) + x0.shape)
invN = BlockDiagonalOperator(invNs, new_axisin=0)
y  = H(x0) + n

A = H.T * invN * H + 0.0005 * I
solution = pcg(A, H.T(invN(y)))
```

The script computes the bottom-right image.

<img src="figures/lena3_orig.png">

<img src="figures/lena3_deconvolved.png">
