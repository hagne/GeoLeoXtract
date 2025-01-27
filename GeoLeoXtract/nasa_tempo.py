#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:21:15 2025

@author: htelg
"""

import xarray as xr
from . import satlab


def open(p2f):
    """Opens a tempo file and formates to be used by GeoLeoExtract."""
    dsbase = xr.open_dataset(p2f)
    dsgl = xr.open_dataset(p2f, group = 'geolocation')
    dspr = xr.open_dataset(p2f, group = 'product')
    dsdqf = xr.open_dataset(p2f, group = 'quality_diagnostic_flags', drop_variables=['lwmask', 'qctest'])
    
    ds =xr.merge([dsgl, dspr, dsdqf])
    
    ds = ds.rename_dims({'mirror_step': 'x', 'xtrack': 'y'})
    ds = ds.assign_coords(time = ds.time, longitude = ds.longitude, latitude = ds.latitude)
    ds = ds.rename_vars({'latitude': 'lat', 'longitude': 'lon', 'dqf': 'DQF'})

    ds.attrs = dsbase.attrs
    ds.attrs


    ds = ds.drop_vars(['latitude_bounds', 'longitude_bounds'])
    ds = ds.transpose('y', 'x')
    si = Tempo_AOD_ALH(ds, product_version=ds.attrs['Algorithm_Version'])
    return si


class Tempo_AOD_ALH(satlab.GeosSatteliteProducts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.valid_qf = [0,1]
        
        if self.product_info['version'] in ['V01']:
            global_qf = [{'high':   [0], 
                          'medium': [1],
                          'low':    [2],
                          'bad':    [3]}]
            self.qf_managment = satlab.QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )

        else:
            raise satlab.GoesExceptionVerionNotRecognized(message = f"Version {self.product_info['version']} not recognized.")
            
    def project_on_sites(self, sites):
        for site in sites: 
            assert(self.ds.attrs['granule_num'] == site['tempo_granule']), f'grnaual_num in file ({self.ds.attrs['granule_num']}) and tempo_granule of site ({site['tempo_granule']}) disagree.'
        return super().project_on_sites(sites)
            
        