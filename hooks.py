import os
import re
import sys
from numpy.distutils.command.build_ext import build_ext
from subprocess import call, Popen, PIPE
from distutils.core import Command
from distutils.command.sdist import sdist as _sdist
from distutils.command.build import build as _build

try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    root = os.path.dirname(os.path.abspath(sys.argv[0]))


def get_version(name, default):
    version = get_version_git()
    if version != '':
        return version
    return get_version_init_file(name) or default


def get_version_git():
    GIT = "git"
    if sys.platform == "win32":
        GIT = "git.cmd"
    stdout, stderr = Popen(
        [GIT, 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE, stderr=PIPE
    ).communicate()
    if stderr != '':
        return ''
    branch = stdout[:-1]
    if re.match('^(v[0-9.]+|master)$', branch) is not None:
        branch = ''
    stdout, stderr = Popen(
        [GIT, 'describe', '--tags', '--dirty', '--abbrev=5'],
        cwd=root,
        stdout=PIPE,
        stderr=PIPE,
    ).communicate()
    if stderr != '':
        raise ValueError('No tag has been found.')
    regex = r"""^
    (?P<tag>.*?)
    (?:-
        (?P<rev>\d+)
        (?P<commit>-g[0-9a-f]{5})
    )?
    (?P<dirty>-dirty)?
    $"""
    m = re.match(regex, stdout.strip(), re.VERBOSE)
    version = m.group('tag')
    if branch != '':
        version += '-' + branch
    if m.group('rev') is not None:
        version += '.dev{0:02}'.format(int(m.group('rev'))) + m.group('commit')
    if m.group('dirty') is not None:
        version += m.group('dirty')
    return version


def get_version_init_file(name):
    try:
        f = open(os.path.join(name, '__init__.py')).read()
    except IOError:
        return ''
    m = re.search(r"__version__ = '(.*)'", f)
    if m is None:
        return ''
    return m.groups()[0]


def write_version(name, version):
    init = open(os.path.join(root, name, '__init__.py.in')).readlines()
    init += ['\n', '__version__ = ' + repr(version) + '\n']
    open(os.path.join(root, name, '__init__.py'), 'w').writelines(init)


class BuildCommand(_build):
    def run(self):
        write_version(self.distribution.get_name(), self.distribution.get_version())
        _build.run(self)


class SDistCommand(_sdist):
    def make_release_tree(self, base_dir, files):
        write_version(self.distribution.get_name(), self.distribution.get_version())
        _sdist.make_release_tree(self, base_dir, files)


class CoverageCommand(Command):
    description = "run the package coverage"
    user_options = [('file=', 'f', 'restrict coverage to a specific file')]

    def run(self):
        call(
            [
                'nosetests',
                '--with-coverage',
                '--cover-package',
                'pyoperators',
                self.file,
            ]
        )
        call(['coverage', 'html'])

    def initialize_options(self):
        self.file = 'test'

    def finalize_options(self):
        pass


class TestCommand(Command):
    description = "run the test suite"
    user_options = [('file=', 'f', 'restrict test to a specific file')]

    def run(self):
        call(['nosetests', self.file])

    def initialize_options(self):
        self.file = 'test'

    def finalize_options(self):
        pass


def get_cmdclass():
    return {
        'build': BuildCommand,
        'build_ext': build_ext,
        'coverage': CoverageCommand,
        'sdist': SDistCommand,
        'test': TestCommand,
    }
