# -*- coding: utf-8 -*-


# on anaconda importing basemap can cause errors which can be fixed by setting
# the following variable
# try:
#     import mpl_toolkits.basemap as __
# except KeyError as er:
#     if er.args[0] == 'PROJ_LIB':
#         import os as _os
#         _os.environ['PROJ_LIB']  = _os.environ['CONDA_PREFIX']
#         import mpl_toolkits.basemap as __
#     else:
#         raise

from . import satlab
from . import cloud_interface
from . import satscraper