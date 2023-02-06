import hooks
import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

ext_modules = [
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
    cmdclass=hooks.cmdclass,
    ext_modules=ext_modules,
)
