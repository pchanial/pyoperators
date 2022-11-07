---
layout: default
title: Installation
excerpt: Install pyoperators package
---

<br>
The PyOperators package is under the
[CECILL-B](http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html)
license, a BSD-like free software license, with a citation requirement.
The source code and development is hosted at
[github](https://github.com/pchanial/pyoperators). The current version
of the pyoperators package is {{ site.version_stable }}.
<br>
<br>

# THIS SECTION IS COMPLETELY OUTDATED.

# Preparation

## The pip utility

The [pip](http://www.pip-installer.org/en/latest/index.html) program
allows the installation of Python packages from the internet. It is
recommended to use it instead of `easy_install`, which cannot uninstall
packages. If `pip` is not installed, but `easy_install` is (or
`easy_install-2.7`), the latter can be used to install the former.
First, check that `easy_install` actually runs the Python interpreter
you're using, by inspecting its first first line of code. Then, type

```bash
sudo easy_install pip
```

## The pyFFTW module

The FFTW package and its Python wrapper need to be installed first.

- on Linux/Ubuntu with root permissions, this will simply be:

```bash
sudo apt-get install libfftw3-dev
sudo pip install pyFFTW
```

- on Linux, without root permissions, one should make sure that the
    following libraries are available: `libfftw3.so`, `libfftw3f.so`,
    `libfftw3l.so`, `libfftw3_threads.so`, `libfftw3f_threads.so`,
    `libfftw3l_threads.so` in a directory `${FFTWPATH}/lib` and that the
    file `fftw3.h` is in `${FFTWPATH}/include`. This [bash
    script](install-fftw.sh) can help for this purpose. Then the wrapper
    can be installed locally:

```bash
MYSOFTDIR=~/software/lib/python2.7/site-packages
CPATH=${FFTWPATH}/include LIBRARY_PATH=${FFTWPATH}/lib pip install
pyFFTW ---install-option="---install-lib=${MYSOFTDIR}"

export PYTHONPATH=${MYSOFTDIR}:${PYTHONPATH}
python -c "import pyfftw; print('Successful installation!')"
```

- on MacOS X, using the MacPorts Python interpreter:

```bash
sudo port install py-pyfftw
```

- on MacOS X, using the default Python interpreter or that from a distribution such as Anaconda or Enthought. Note that depending on your installation, a `sudo` might be required.

With MacPort:
```bash
sudo port install fftw-3 fftw-3-long fftw-3-single
CPATH=/opt/local/include LIBRARY_PATH=/opt/local/lib pip install
pyFFTW
```

With HomeBrew:
```bash
brew install fftw
CPATH=/usr/local/include LIBRARY_PATH=/usr/local/lib pip install
pyFFTW
```

# Installation

## The easy way

The easiest way to download and install the PyOperators package is by
using pip. It will fetch the package and its dependencies from the
[Python Package Index](http://pypi.python.org/pypi) and install them.

```bash
sudo pip install pyoperators
```

To upgrade an already installed PyOperators package:

```bash
sudo pip install ---upgrade pyoperators
```

## The Git way

You can also clone the Github repository, using the version control
system [Git](http://git-scm.com/):

```bash
git clone git://github.com/pchanial/pyoperators
sudo pip install ./pyoperators
```

This method is convenient to test PyOperators' development branch but
package dependencies must be installed beforehand (see the
[setup.py](https://github.com/pchanial/pyoperators/blob/v{{site.version_stable}}/setup.py)
file for the list of requirements). The package can be updated by typing
(if you've just cloned the repository, you will obviously not get
anything new):

```bash
git pull
```

```text
Already up-to-date.
```
