import hooks
import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

ext_modules = [
    Extension(
        'pyoperators.utils.cythonutils',
        sources=['pyoperators/utils/cythonutils.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'pyoperators.utils.ufuncs',
        sources=['pyoperators/utils/ufuncs.c.src'],
    ),
]

setup(
    cmdclass=hooks.cmdclass,
    ext_modules=ext_modules,
)
