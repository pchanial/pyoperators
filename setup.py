import hooks
import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

VERSION = '0.13'

name = 'pyoperators'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

ext_modules = [
    Extension(
        'pyoperators.utils.cythonutils',
        sources=['pyoperators/utils/cythonutils.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'pyoperators.utils.ufuncs',
        sources=["pyoperators/utils/ufuncs.c.src"],
    ),
]

setup(
    name=name,
    version=hooks.get_version(name, VERSION),
    description='Operators and solvers for high-performance computing.',
    long_description=long_description,
    url='http://pchanial.github.com/pyoperators',
    author='Pierre Chanial',
    author_email='pierre.chanial@gmail.com',
    maintainer='Pierre Chanial',
    maintainer_email='pierre.chanial@gmail.com',
    setup_requires=['numpy'],
    install_requires=['numexpr>=2', 'numpy>=1.6', 'scipy>=0.9'],
    extras_require={
        'fft': ['pyfftw'],
        'mpi': ['mpi4py'],
        'wavelet': ['pywt'],
    },
    packages=['pyoperators', 'pyoperators.iterative', 'pyoperators.utils'],
    platforms=platforms.split(','),
    keywords=keywords.split(','),
    cmdclass=hooks.cmdclass,
    ext_modules=ext_modules,
    license='CeCILL-B',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
)
