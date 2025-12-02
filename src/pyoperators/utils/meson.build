# Install Python sources for utils subpackage
py.install_sources(
  [
    '__init__.py',
    'fake_MPI.py',
    'misc.py',
    'mpi.py',
    'testing.py',
  ],
  subdir: 'pyoperators/utils'
)

# Extension 1: cythonutils (Cython extension)
cythonutils_source = custom_target(
  'cythonutils_c',
  output: 'cythonutils.c',
  input: 'cythonutils.pyx',
  command: [
    py, '-m', 'cython',
    '-3',
    '--output-file', '@OUTPUT@',
    '@INPUT@'
  ]
)

py.extension_module(
  'cythonutils',
  cythonutils_source,
  c_args: numpy_nodepr_api,
  dependencies: [py_dep, numpy_inc],
  install: true,
  subdir: 'pyoperators/utils'
)

# Extension 2: ufuncs (C extension)
py.extension_module(
  'ufuncs',
  'ufuncs.c',
  c_args: numpy_nodepr_api,
  dependencies: [py_dep, numpy_inc],
  install: true,
  subdir: 'pyoperators/utils'
)
