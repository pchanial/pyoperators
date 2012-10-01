import os

VERSION = '0.7-dev'

LOCAL_PATH = os.path.join(os.path.expanduser('~'), '.local/share/pyoperators')
if not os.path.exists(LOCAL_PATH):
    os.makedirs(LOCAL_PATH)

del os
