#!/usr/bin/env python
import numpy as np
import re
import subprocess
import sys
from distutils.extension import Extension
from numpy.distutils.core import setup, Command
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import get_info
from subprocess import Popen, PIPE

VERSION = '0.10'


def version_sdist():
    stdout, stderr = Popen(['git', 'tag', '--contains', 'HEAD'],
                           stdout=PIPE, stderr=PIPE).communicate()
    # use commit tag
    if stderr:
        return VERSION
    tag = stdout.split('\n')[0]
    if tag != '':
        if tag[0] == 'v':
            tag = tag[1:]
        return tag

    # use version
    version = VERSION

    # use branch unless it is 'master' or e.g. 'v3.4'
    stdout, stderr = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                           stdout=PIPE, stderr=PIPE).communicate()
    if stderr:
        return version
    branch = stdout[:-1]
    if re.match('(v[0-9.]+|master)', branch) is None:
        version += '-' + branch

    # use commit's short hash
    stdout, stderr = Popen(['git', 'rev-parse', '--verify', '--short', 'HEAD'],
                           stdout=PIPE, stderr=PIPE).communicate()
    if stderr:
        return version
    return version + '-' + stdout[:-1]


class NewCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class CoverageCommand(NewCommand):
    description = "run the package coverage"

    def run(self):
        subprocess.call(['nosetests', '--with-coverage', '--cover-package',
                         'pyoperators'] + coverage_extra)
        subprocess.call(['coverage', 'html'])


class TestCommand(NewCommand):
    description = "run the test suite"

    def run(self):
        subprocess.call(['nosetests', 'test'])

if 'coverage' in sys.argv:
    index = sys.argv.index('coverage') + 1
    coverage_extra = sys.argv[index:]
    sys.argv = sys.argv[:index]

version = version_sdist()
print 'VERSION:', version

if 'install' in sys.argv[1:]:
    if '-' in version:
        version = VERSION + '-dev'

if any(c in sys.argv[1:] for c in ('install', 'sdist')):
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
                'pyfftw'],
      install_requires=['numexpr>2'],
      packages=['pyoperators', 'pyoperators.iterative', 'pyoperators.utils'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass={'build_ext': build_ext,
                'coverage': CoverageCommand,
                'test': TestCommand},
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
          'Topic :: Scientific/Engineering'])
