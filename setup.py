#!/usr/bin/env python
from distutils.core import setup

def version():
    import os, re
    f = open(os.path.join('pyoperators', '__init__.py')).read()
    m = re.search(r"__version__ = '(.*)'", f)
    return m.groups()[0]

version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

setup(name='pyoperators',
      version=version,
      description='Operators and solvers for high-performance computing.',
      long_description=long_description,
      url='http://pchanial.github.com/pyoperators',
      author='Pierre Chanial & Nicolas Barbey',
      author_email='pierre.chanial@gmail.com & nicolas.a.barbey@gmail.com',
      maintainer='Pierre Chanial',
      maintainer_email='pierre.chanial@gmail.com',
      requires=['numpy (>1.6)', 'scipy (>0.9)', 'numexpr (>2.0)'],
      packages=['pyoperators', 'pyoperators.iterative'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      license='CeCILL-B',
      classifiers = [
          "Programming Language :: Python",
          "Programming Language :: Python :: 2 :: Only",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering"])
