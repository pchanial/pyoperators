#!/usr/bin/env python
import numpy as np
import hooks
from hooks import get_extension, get_cmdclass, get_version
from numpy.distutils.core import setup

VERSION = '0.12'

hooks.RECOMPILE_CYTHON = False

name = 'pyoperators'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

ext_modules = [
    get_extension(
        "pyoperators.utils.cythonutils",
        sources=["pyoperators/utils/cythonutils.pyx"],
        include_dirs=[np.get_include()],
    ),
    get_extension(
        "pyoperators.utils.ufuncs", sources=["pyoperators/utils/ufuncs.c.src"]
    ),
]

setup(
    name=name,
    version=get_version(name, VERSION),
    description='Operators and solvers for high-performance computing.',
    long_description=long_description,
    url='http://pchanial.github.com/pyoperators',
    author='Pierre Chanial',
    author_email='pierre.chanial@gmail.com',
    maintainer='Pierre Chanial',
    maintainer_email='pierre.chanial@gmail.com',
    requires=['numpy(>=1.6)', 'scipy(>=0.9)', 'pyfftw'],
    install_requires=['numexpr>2'],
    packages=['pyoperators', 'pyoperators.iterative', 'pyoperators.utils'],
    platforms=platforms.split(','),
    keywords=keywords.split(','),
    cmdclass=get_cmdclass(),
    ext_modules=ext_modules,
    license='CeCILL-B',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
)
