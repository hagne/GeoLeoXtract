#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:21:15 2025

@author: htelg
"""

import xarray as xr
import pandas as _pd

from . import satlab
from . import file_io


def open(p2f):
    """Opens a tempo file and formates to be used by GeoLeoExtract."""
    dsbase = xr.open_dataset(p2f)
    dsgl = xr.open_dataset(p2f, group = 'geolocation')
    dspr = xr.open_dataset(p2f, group = 'product')
    dsdqf = xr.open_dataset(p2f, group = 'quality_diagnostic_flags', drop_variables=['lwmask', 'qctest'])
    
    ds =xr.merge([dsgl, dspr, dsdqf])
    
    ds = ds.rename_dims({'mirror_step': 'x', 'xtrack': 'y'})
    ds = ds.assign_coords(time = ds.time, longitude = ds.longitude, latitude = ds.latitude)
    ds = ds.rename_vars({'latitude': 'lat', 'longitude': 'lon', 'dqf': 'DQF', 
                         'time': 'overpass_time',
                         })

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
            assert(self.ds.attrs['granule_num'] == site['tempo_granule']), f'grnaual_num in file ({self.ds.attrs["granule_num"]}) and tempo_granule of site ({site["tempo_granule"]}) disagree.'
        return super().project_on_sites(sites)
            
def project_statellite2stations_v01(path2file_in, stations, path2file_out = None, test = False, verbose = False):
    """
    Version 1. Projects satellite data to specific coordinates and their surrounding.

    Parameters
    ----------
    row : TYPE
        DESCRIPTION.
    stations : TYPE
        DESCRIPTION.
    path2file_out: 
        Path to save netcdf. If None noting is saved, but dataset is still 
        returned.

    Returns
    -------
    None.

    """

    # read the file
    ngsinst = file_io.open_file(path2file_in)
    if test == 1:
        return ngsinst
    # project to stations
    projection = ngsinst.project_on_sites(stations)
    if test == 2:
        return projection
    # merge closest gridpoint and area
    point = projection.projection2point.copy()#.sel(site = 'TBL')
    point['DQF'] = point.DQF.astype(int) # for some reason this was float64... because there are some nans in there
    
    # change var names to distinguish from area
    if verbose:
        print(f'ngsinst.valid_2D_variables: {ngsinst.valid_2D_variables}')
    for var in ngsinst.valid_2D_variables:
        point = point.rename({var: f'{var}_on_pixel',})
        if f'{var}_DQF_assessed' in point.variables:
            point = point.rename({f'{var}_DQF_assessed': f'{var}_on_pixel_DQF_assessed',})
    point = point.rename({'DQF': 'DQF_on_pixel'})
    if test == 3:
        return projection
    # merge aerea and point
    ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
    # add a time stamp
    ds = ds.merge(projection.overpass_times)
    # remove site since I process by site
    ds = ds.sel(site = str(ds.site.values[0])).drop_vars('site')
    ds = ds.expand_dims({'time': [ds.overpass_time.values]})
    
    
    
    # global attribute
    ds.attrs['info'] = ('This file contains a projection of satellite data onto specific sites.\n'
                         'It includes the closest pixel data as well as the average over circular\n'
                         'areas with various radii. Note, for the averaged data only data is\n'
                         'considered with a qulity flag given by the prooduct class in the\n'
                         'GeoLeoXtract library.')
    

    # save2file
    if test == 4:
        return ds, projection
    if not isinstance(path2file_out, type(None)):
        ds.to_netcdf(path2file_out)
        # Memory kept on piling up -> maybe a cleanup will help
        ds.close()
    ngsinst.ds.close()
    ngsinst = None
    return ds
    
    