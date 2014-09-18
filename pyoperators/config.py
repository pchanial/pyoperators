import os
import site
from .warnings import warn, PyOperatorsWarning


def getenv(key):
    val = os.getenv(key, '').strip()
    if len(val) == 0:
        return False
    try:
        val = int(val)
    except ValueError:
        warn("Invalid environment variable {0}='{1}'".format(key, val))
        return False
    return bool(val)


LOCAL_PATH = os.getenv('PYOPERATORSPATH')
if LOCAL_PATH is None:
    LOCAL_PATH = os.path.join(site.USER_BASE, 'share', 'pyoperators')
if not os.path.exists(LOCAL_PATH):
    try:
        os.makedirs(LOCAL_PATH)
    except IOError:
        warn(
            "User path '{0}' cannot be created.".format(LOCAL_PATH), PyOperatorsWarning
        )
elif not os.access(LOCAL_PATH, os.W_OK):
    warn("User path '{0}' is not writable.".format(LOCAL_PATH), PyOperatorsWarning)

PYOPERATORS_NO_MPI = getenv('PYOPERATORS_NO_MPI')
PYOPERATORS_VERBOSE = getenv('PYOPERATORS_VERBOSE')

del os, site, PyOperatorsWarning, warn, getenv
