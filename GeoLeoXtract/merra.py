#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:01:27 2025

@author: htelg
"""
import xarray as _xr
import pandas as _pd
from . import satlab 

def open_M2T1NXAER(p2f):
    ds = _xr.open_dataset(p2f)
    ds = ds.rename_dims(time = 'datetime')
    ds = ds.rename_vars(time = 'datetime')
    ds['DQF'] = _xr.zeros_like(ds[max(ds.data_vars, key=lambda var: len(ds[var].dims))])
    si = M2T1NXAER(ds, product_version = ds.attrs['VersionID'])
    return si
    
    
class M2T1NXAER(satlab.GeosSatteliteProducts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.valid_qf = [0,1]
        
        if self.product_info['version'] in ['5.12.4',]:
            
            global_qf = [{'high':   [0],}]
            self.qf_managment = satlab.QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
        # elif self.product_info['version'] in ['M3',]:
        #     global_qf = [{'high':   [0], 
        #                   'low': [1],
        #                   }]
        #     self.qf_managment = QfManagment(self, 
        #                                     qf_representation='as_is', 
        #                                     global_qf= global_qf, 
        #                                    )
        else:
            raise satlab.GoesExceptionVerionNotRecognized(message = f"Version {self.product_info['version']} not recognized.")
            
            
def make_product_v01(M2T1NXAER_instance, stations, path2file_out = None, test = False, verbose = False):
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

    ngsinst = M2T1NXAER_instance
    # project to stations
    projection = ngsinst.project_on_sites(stations)
    projection.radii = [25, 50, 100, 200]

    if test:
        return projection
    # merge closest gridpoint and area
    point = projection.projection2point.copy()#.sel(site = 'TBL')
    point['DQF'] = point.DQF.astype(int) # for some reason this was float64... because there are some nans in there
    
    # change var names to distinguish from area
    if verbose:
        print(f'ngsinst.valid_2D_variables: {ngsinst.valid_2D_variables}')
    for var in ngsinst.valid_2D_variables:
        point = point.rename({var: f'{var}_on_pixel',})
        # if f'{var}_DQF_assessed' in point.variables:
        #     point = point.rename({f'{var}_DQF_assessed': f'{var}_on_pixel_DQF_assessed',})
    point = point.rename({'DQF': 'DQF_on_pixel'})
    


    if test:
        return projection
    # merge aerea and point
    ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
    
    #### Drop data quality as there is only one
    ds = ds.isel(data_quality = 0)
    ds = ds.drop_vars('data_quality')

    # global attribute
    ds.attrs['info'] = ('This file contains a projection of satellite data onto specific sites.\n'
                         'It includes the closest pixel data as well as the average over circular\n'
                         'areas with various radii. Note, for the averaged data only data is\n'
                         'considered with a qulity flag given by the prooduct class in the\n'
                         'GeoLeoXtract library.')
    

    # save2file
    if test:
        return ds
    if not isinstance(path2file_out, type(None)):
        ds.to_netcdf(path2file_out)
        # Memory kept on piling up -> maybe a cleanup will help
        ds.close()
    ngsinst.ds.close()
    ngsinst = None
    
    return ds