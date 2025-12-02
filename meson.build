project('pyoperators',
  'c', 'cython',
  version: run_command('python', '-c',
    'from setuptools_scm import get_version; print(get_version())',
    check: false).stdout().strip(),
  meson_version: '>= 1.1.0',
  default_options: [
    'warning_level=2',
  ],
)

py = import('python').find_installation(pure: false)

# Check Python version
if py.language_version().version_compare('< 3.10')
  error('Python 3.10 or newer is required')
endif

# Dependencies
py_dep = py.dependency()

# Get NumPy include directory
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# Create a NumPy dependency with system includes (to avoid Meson absolute path error)
numpy_inc = declare_dependency(
  compile_args: ['-I' + incdir_numpy]
)

# Compiler options
cc = meson.get_compiler('c')

# Define NPY_NO_DEPRECATED_API macro for NumPy 2.0 compatibility
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION']

# Add subdirectories
subdir('src')
