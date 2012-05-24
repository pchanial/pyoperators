#!/usr/bin/env python
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def version():
    import os, re
    f = open(os.path.join('pyoperators', '__init__.py')).read()
    m = re.search(r"__version__ = '(.*)'", f)
    return m.groups()[0]

version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

ext_modules = [Extension("pyoperators.utils.cythonutils",
                         sources=["pyoperators/utils/cythonutils.pyx"],
                         include_dirs=['.', np.get_include()],
                         )]

setup(name='pyoperators',
      version=version,
      description='Operators and solvers for high-performance computing.',
      long_description=long_description,
      url='http://pchanial.github.com/pyoperators',
      author='Pierre Chanial',
      author_email='pierre.chanial@gmail.com',
      maintainer='Pierre Chanial',
      maintainer_email='pierre.chanial@gmail.com',
      requires=['numpy (>1.6)', 'scipy (>0.9)', 'numexpr (>2.0)'],
      packages=['pyoperators', 'pyoperators.iterative', 'pyoperators.utils'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      license='CeCILL-B',
      classifiers = [
          "Programming Language :: Python",
          "Programming Language :: Python :: 2 :: Only",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering"])
