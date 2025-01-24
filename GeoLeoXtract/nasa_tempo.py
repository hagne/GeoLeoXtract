#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:21:15 2025

@author: htelg
"""

import xarray as xr
import GeoLeoXtract.satlab as glx_satlab


def open(p2f):
    """Opens a tempo file and formates to be used by GeoLeoExtract."""
    dsbase = xr.open_dataset(p2f)
    dsgl = xr.open_dataset(p2f, group = 'geolocation')
    dspr = xr.open_dataset(p2f, group = 'product')
    ds =xr.merge([dsgl, dspr])
    
    ds = ds.rename_dims({'mirror_step': 'x', 'xtrack': 'y'})
    ds = ds.assign_coords(time = ds.time, longitude = ds.longitude, latitude = ds.latitude)
    ds = ds.rename_vars({'latitude': 'lat', 'longitude': 'lon'})

    ds.attrs = dsbase.attrs
    ds.attrs


    ds = ds.drop_vars(['latitude_bounds', 'longitude_bounds'])
    ds = ds.transpose('y', 'x')
    si = glx_satlab.GeosSatteliteProducts(ds)
    return si