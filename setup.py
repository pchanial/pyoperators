#!/usr/bin/env python
from distutils.core import setup
setup(name='operators',
      version='0.0.0',
      description='Non-Linear Operators, Linear Operators and Iterative algorithms',
      author='',
      author_email='',
      requires=['numpy', 'scipy', ],
      #packages=['operators', 'operators.wrappers', 'operators.iterative'],
      packages=['operators', 'operators.iterative']
      )
