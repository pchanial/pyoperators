import os

VERSION = '0.6-dev'

LOCAL_PATH = os.path.join(os.path.expanduser('~'), '.local/share')
if os.path.exists(LOCAL_PATH):
    LOCAL_PATH = os.path.join(LOCAL_PATH, 'pyoperators')
else:
    LOCAL_PATH = os.path.join(os.path.expanduser('~'), '.pyoperators')
try:
    os.mkdir(LOCAL_PATH)
except OSError:
    pass

del os
