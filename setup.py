#!/usr/bin/env python
import numpy as np
from distutils.extension import Extension
from numpy.distutils.core import setup
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import get_info

def version():
    import os, re
    f = open(os.path.join('pyoperators', 'config.py')).read()
    m = re.search(r"VERSION = '(.*)'", f)
    return m.groups()[0]

version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

ext_modules = [Extension("pyoperators.utils.cythonutils",
                         sources=["pyoperators/utils/cythonutils.c"],
                         include_dirs=['.', np.get_include()],
                         ),
               Extension("pyoperators.utils.ufuncs",
                         sources=["pyoperators/utils/ufuncs.c.src"],
                         extra_info=get_info("npymath"))]

setup(name='pyoperators',
      version=version,
      description='Operators and solvers for high-performance computing.',
      long_description=long_description,
      url='http://pchanial.github.com/pyoperators',
      author='Pierre Chanial',
      author_email='pierre.chanial@gmail.com',
      maintainer='Pierre Chanial',
      maintainer_email='pierre.chanial@gmail.com',
      install_requires=['numpy>=1.6',
                        'numexpr>2',
                        'pyfftw',
                        'scipy>=0.9',
                ],
      packages=['pyoperators', 'pyoperators.iterative', 'pyoperators.utils'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      license='CeCILL-B',
      classifiers = [
          'Programming Language :: Python',
          'Programming Language :: Python :: 2 :: Only',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering'])
