import os
import site

VERSION = '0.7-dev'

LOCAL_PATH = os.path.join(site.USER_BASE, 'share', 'pyoperators')
if not os.path.exists(LOCAL_PATH):
    os.makedirs(LOCAL_PATH)

del os
