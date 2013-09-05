#!/usr/bin/env python
import numpy as np
import re
import sys
from distutils.extension import Extension
from numpy.distutils.core import setup
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import get_info
from subprocess import Popen, PIPE

VERSION = '0.8'


def version_sdist():
    p = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE,
              stderr=PIPE)
    stdout, stderr = p.communicate()
    if stderr:
        return VERSION
    branch = stdout[:-1]
    if re.search('^v[0-9]', branch) is not None:
        branch = branch[1:]
    if branch != 'master':
        return VERSION
    p = Popen(['git', 'rev-parse', '--verify', '--short', 'HEAD'], stdout=PIPE,
              stderr=PIPE)
    stdout, stderr = p.communicate()
    version = VERSION
    if not stderr:
        version += '-' + stdout[:-1]
    return version

version = version_sdist()
if 'install' in sys.argv[1:]:
    if '-' in version:
        version = VERSION + '-dev'

if (c in sys.argv[1:] for c in ('install', 'sdist')):
    init = open('pyoperators/__init__.py.in').readlines()
    init += ['\n', '__version__ = ' + repr(version) + '\n']
    open('pyoperators/__init__.py', 'w').writelines(init)

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
      requires=['numpy(>=1.6)',
                'scipy(>=0.9)',
                'pyfftw',
               ],
      install_requires=['numexpr>2'],
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
