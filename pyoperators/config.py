import os as _os
import site as _site

from .warnings import PyOperatorsWarning as _PyOperatorsWarning
from .warnings import warn as _warn


def getenv(key, default, cls):
    val = _os.getenv(key, '').strip()
    if len(val) == 0:
        return cls(default)
    try:
        if cls is bool:
            val = int(val)
        val = cls(val)
    except ValueError:
        _warn(f"Invalid environment variable {key}='{val}'", _PyOperatorsWarning)
        return cls(default)
    return val


# PyOperators local path, used for example to store the FFTW wisdom files
LOCAL_PATH = _os.getenv('PYOPERATORS_PATH')
if LOCAL_PATH is None:
    LOCAL_PATH = _os.path.join(_site.USER_BASE, 'share', 'pyoperators')
if not _os.path.exists(LOCAL_PATH):
    try:
        _os.makedirs(LOCAL_PATH)
    except OSError:
        _warn(f"User path '{LOCAL_PATH}' cannot be created.", _PyOperatorsWarning)
elif not _os.access(LOCAL_PATH, _os.W_OK):
    _warn(f"User path '{LOCAL_PATH}' is not writable.", _PyOperatorsWarning)

# force garbage collection when deleted operators' nbytes exceed this
# threshold.
GC_NBYTES_THRESHOLD = getenv('PYOPERATORS_GC_NBYTES_THRESHOLD', 1e8, float)

MEMORY_ALIGNMENT = getenv('PYOPERATORS_MEMORY_ALIGNMENT', 32, int)

# We allow reuse of pool variables only if they do not exceed 20% of
# the requested size
MEMORY_TOLERANCE = getenv('PYOPERATORS_MEMORY_TOLERANCE', 1.2, float)

# on some supercomputers, importing mpi4py on a login node exits python without
# raising an ImportError.
NO_MPI = getenv('PYOPERATORS_NO_MPI', False, bool)

VERBOSE = getenv('PYOPERATORS_VERBOSE', False, bool)

del getenv
