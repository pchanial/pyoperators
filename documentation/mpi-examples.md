---
layout: post
title: 6. MPI examples
category: doc
---

6.1 <a name="distributionidentityoperator"></a> Using MPIDistributionIdentityOperator
-------------------------------------------------------------------------------------

In this example, we will tell each MPI process to randomly and
independly observe 20% of an input image and collect all this
information to reconstruct the input image. Each observation is assumed
to be affected by an independent gaussian noise.

```python
import numpy as np
import scipy
from mpi4py import MPI
from pyoperators import MPIDistributionIdentityOperator, HomothetyOperator, MaskOperator, pcg

x0 = scipy.misc.lena().astype(float)
sigma = 2.
n = np.random.standard_normal(x0.shape) * sigma
invN = HomothetyOperator(1 / sigma**2)

H = MaskOperator(np.random.random(x0.shape) > 0.2) * \
    MPIDistributionIdentityOperator()
y = H(x0) + n

A = H.T * invN * H
solution = pcg(A, H.T(invN(y)))

if MPI.COMM_WORLD.rank == 0:
    np.save('mpi_n{}.npy'.format(MPI.COMM_WORLD.size), solution['x'])
```

The `MPIDistributionIdentityOperator` is a block column operator whose
blocks are the identity and for which the output of each block is
handled by an MPI process according to its rank. Hence the MPI reduction
is performed by the transpose of this operator (present in `H.T`): a
block row operator whose blocks are the identity.  
Results for 1, 2, 4 and 8 MPI processes are shown in the following
figure.

<hr>

<img src="{{site.baseurl}}/documentation/figures/lena_mpi1.png">

6.2 <a name="distributionglobaloperator"></a> Using MPIDistributionGlobalOperator
---------------------------------------------------------------------------------

This operator takes a global input and scatter it as local outputs.
Currently only one scattering scheme is implemented: the input is split
along its first dimension by equally distributing it by increasing rank.

<div class="definition">

Given an input of shape (nglobal, …) and p MPI processes, the process of
rank r will be handed an output of shape (nlocal, …) with nlocal given
by

<div class="centered">

`nlocal = nglobal // p + ((nglobal % p) > r)`

</div>
</div>

This operator is of limited interest for inverse problems, since the
local sections are completely independent. It can still be a handy way
to distribute some work. The following example takes an RGB image, split
it by color and save the monochromatic image convolved by a
color-dependent kernel.

```python
# script.py
from PIL import Image
import numpy as np
import matplotlib.cbook as cbook
from mpi4py import MPI
from pyoperators import ConvolutionOperator, MPIDistributionGlobalOperator, ReshapeOperator

datafile = cbook.get_sample_data('lena.jpg')
image = Image.open(datafile)
image.load()
x = np.rollaxis(np.asarray(image, float), 2)
global_shape = x.shape          # (3,512,512)
local_shape = global_shape[1:]  # (512,512)

color = ('red', 'green', 'blue')[MPI.COMM_WORLD.rank]
kernel_size = {'red':11, 'green':7, 'blue':3}[color]
kernel = np.ones(2*(kernel_size,)) / kernel_size**2

H = ConvolutionOperator(kernel, local_shape)         * \
    ReshapeOperator((1,) + local_shape, local_shape) * \
    MPIDistributionGlobalOperator(global_shape)

y = H(x)
np.save('mpi_{}.npy'.format(color), y)
```

This script only runs with 3 MPI processes:

```bash
$ mpirun -n 3 python script.py
```
<img src="{{ site.baseurl}}/documentation/figures/lena_mpi2.png" style="width:80%">