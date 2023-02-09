import numpy as np
from Cython.Build import cythonize
from numpy.distutils.core import setup  # for the pre-processing of .c.src files
from setuptools import Extension

extensions = [
    Extension(
        'pyoperators.utils.cythonutils',
        sources=['src/pyoperators/utils/cythonutils.pyx'],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        'pyoperators.utils.ufuncs',
        sources=['src/pyoperators/utils/ufuncs.c.src'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
]

setup(
    ext_modules=cythonize(extensions),
)
