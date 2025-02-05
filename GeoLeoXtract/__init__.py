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
from . import products
from . import processing
from . import file_io_hrrr
from . import projector

from .file_io import open_file
# from . import config as _config

# config = _config.load_config()

import json as _json
import pathlib as _pl

class Config():
    def __init__(self):
        self._values = None
        return

    @property
    def values(self):
        if isinstance(self._values, type(None)):
# def _load_config(path2file='~/.GeoLeoXtract', verbose = True):
            path2file='~/.GeoLeoXtract'
            filename = _pl.Path(path2file)
            filename = filename.expanduser()
            if filename.is_file():
                with open(filename, 'r') as file:
                    self._values = _json.load(file)
            else:
                msg = f"""Configuration file not found          
Please create  {filename}\n"""
                msg += """Example file content:
{
    "earthdata_credentials": {
        "username": "yourusername",
        "password": "your_password",
    },
}"""
                
                assert(False), msg
                
        return self._values

config = Config()