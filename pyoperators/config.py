import os
import site
from .warnings import warn, PyOperatorsWarning


def getenv(key, default, cls):
    val = os.getenv(key, '').strip()
    if len(val) == 0:
        return cls(default)
    try:
        if cls is bool:
            val = int(val)
        val = cls(val)
    except ValueError:
        warn("Invalid environment variable {0}='{1}'".format(key, val),
             PyOperatorsWarning)
        return cls(default)
    return val


# PyOperators local path, used for example to store the FFTW wisdom files
LOCAL_PATH = os.getenv('PYOPERATORS_PATH')
if LOCAL_PATH is None:
    LOCAL_PATH = os.path.join(site.USER_BASE, 'share', 'pyoperators')
if not os.path.exists(LOCAL_PATH):
    try:
        os.makedirs(LOCAL_PATH)
    except IOError:
        warn("User path '{0}' cannot be created.".format(LOCAL_PATH),
             PyOperatorsWarning)
elif not os.access(LOCAL_PATH, os.W_OK):
    warn("User path '{0}' is not writable.".format(LOCAL_PATH),
         PyOperatorsWarning)

# force garbage collection when deleted operators' nbytes exceed this
# threshold.
GC_NBYTES_THRESHOLD = getenv('PYOPERATORS_GC_NBYTES_THRESHOLD', 1e8, float)

MEMORY_ALIGNMENT = getenv('PYOPERATORS_MEMORY_ALIGNMENT', 32, int)

# We allow reuse of pool variables only if they do not exceed 20% of
# the requested size
MEMORY_TOLERANCE = getenv('PYOPERATORS_MEMORY_TOLERANCE', 1.2, float)

NO_MPI = getenv('PYOPERATORS_NO_MPI', False, bool)
VERBOSE = getenv('PYOPERATORS_VERBOSE', False, bool)

#del os, site, PyOperatorsWarning, warn, getenv
