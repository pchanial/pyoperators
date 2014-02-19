import os
import site
from .warnings import warn, PyOperatorsWarning

LOCAL_PATH = os.getenv('PYOPERATORSPATH')
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

del os, site, PyOperatorsWarning, warn
