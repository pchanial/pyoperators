#!/usr/bin/env python
from distutils.core import setup
setup(name='pyoperators',
      version='0.0.0',
      description='Non-Linear Operators, Linear Operators and Iterative algorithms',
      author='',
      author_email='',
      requires=['numpy', 'scipy', ],
      packages=['pyoperators', 'pyoperators.iterative']
      )
