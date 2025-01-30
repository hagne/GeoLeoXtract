#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:01:27 2025

@author: htelg
"""
import xarray as _xr
from . import satlab 

def open_M2T1NXAER(p2f):
    ds = _xr.open_dataset(p2f)
    ds = ds.rename_dims(time = 'datetime')
    ds = ds.rename_vars(time = 'datetime')
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