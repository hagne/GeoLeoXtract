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
from . import scrapers

# from . import config as _config

# config = _config.load_config()

import json as _json
import pathlib as _pl

def _load_config(path2file='~/.GeoLeoXtract', verbose = True):
    filename = _pl.Path(path2file)
    filename = filename.expanduser()
    if filename.is_file():
        with open(filename, 'r') as file:
            config = _json.load(file)
    else:
        if verbose:
            print(f'File {filename} not found.')
        config = None
    return config

config = _load_config()