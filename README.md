# PyOperators

The PyOperators package defines operators and solvers for high-performance computing. These operators are multi-dimensional functions with optimised and controlled memory management. If linear, they behave like matrices with a sparse storage footprint.


## Documentaion

https://pchanial.github.io/pyoperators


## Installation

```bash
pip install pyoperators[fft,wavelets]
```

On some platforms, it might be more convenient to install pyfftw through Conda beforehand to use the `FFTOperator`:
```bash
conda install pyfftw
```

For MPI communication, an MPI library needs to be installed, for example on Ubuntu:
```bash
sudo apt install libopenmpi-dev
pip install pyoperators[fft,wavelets,mpi]
```


## Getting started

To define an operator, one needs to define a direct function
which will replace the usual matrix-vector operation:

```python
>>> def f(x, out):
...     out[...] = 2 * x
```

Then, you can instantiate an `Operator`:

```python
>>> A = pyoperators.Operator(direct=f, flags='symmetric')
```

An alternative way to define an operator is to define a subclass:

```python
>>> from pyoperators import flags, Operator
... @flags.symmetric
... class MyOperator(Operator):
...     def direct(x, out):
...         out[...] = 2 * x
...
... A = MyOperator()
```

This operator does not have an explicit shape, it can handle inputs of any shape:

```python
>>> A(np.ones(5))
array([ 2.,  2.,  2.,  2.,  2.])
>>> A(np.ones((2,3)))
array([[ 2.,  2.,  2.],
       [ 2.,  2.,  2.]])
```

By setting the `symmetric` flag, we ensure that A's transpose is A:

```python
>>> A.T is A
True
```

For non-explicit shape operators, we get the corresponding dense matrix by specifying the input shape:

```python
>>> A.todense(shapein=2)
array([[2, 0],
       [0, 2]])
```

Operators do not have to be linear. Many operators are already [predefined](http://pchanial.github.io/pyoperators/2000/doc-operators/#list), such as the `DiagonalOperator`, the `FFTOperator` or the nonlinear `ClipOperator`.

The previous `A` matrix could be defined more easily like this:

```python
>>> from pyoperators import I
>>> A = 2 * I
```

where `I` is the identity operator with no explicit shape.

Operators can be combined together by addition, element-wise multiplication or composition. Note that the operator `*` stands for matrix multiplication if the two operators are linear, or for element-wise multiplication otherwise:

```python
>>> from pyoperators import I, DiagonalOperator
>>> B = 2 * I + DiagonalOperator(range(3))
>>> B.todense()
array([[2, 0, 0],
       [0, 3, 0],
       [0, 0, 4]])
```

Algebraic rules can easily be attached to operators. They are used to simplify expressions to speed up their execution. The `B` Operator has been reduced to:

```python
>>> B
DiagonalOperator(array([2, ..., 4], dtype=int64), broadcast='disabled', dtype=int64, shapein=3, shapeout=3)
```

Many simplifications are available. For instance:

```python
>>> from pyoperators import Operator
>>> C = Operator(flags='idempotent,linear')
>>> C * C is C
True
>>> D = Operator(flags='involutary')
>>> D(D)
IdentityOperator()
```


## Requirements

- Python >= 3.10
- NumPy >= 2.0
- SciPy >= 1.10

Optional requirements:

- PyWavelets: wavelet transforms
- pyfftw: Fast Fourier transforms
- mpi4py: For MPI communication

## Development

To build from source, you'll need:

- Meson >= 1.1.0
- Ninja
- Cython >= 0.29.30
- A C compiler

Build the project:

```bash
pip install -e . --no-build-isolation
```

Or build a wheel:

```bash
python -m build
```
