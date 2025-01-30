#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:34:15 2025

@author: htelg
"""

import xarray as _xr
import numpy as _np
import collections as _collections
import pyproj as _pyproj
import re as  _re
import pandas as _pd
from .opt_imports import pyhdf as _pyhdf

from . import _pylab

def read_Modis_MCD19A2(hdf):
    """
    Parameters
    -----------
    hdf: pyhdf object or path to file
    """
    def parse_metadata(text):
        '''This is a more elegant metadat parser than the other one... implement this one for other uses'''
        # Pattern to capture group and object blocks
        group_pattern = _re.compile(r"GROUP\s+=\s+(\S+)(.*?)END_GROUP\s+=\s+\1", _re.DOTALL)
        object_pattern = _re.compile(r"OBJECT\s+=\s+(\S+)(.*?)END_OBJECT\s+=\s+\1", _re.DOTALL)
        key_value_pattern = _re.compile(r"(\S+)\s+=\s+\"(.*?)\"")
        
        def parse_block(block):
            # Recursively parse object blocks
            parsed = {}
            for obj_match in object_pattern.finditer(block):
                obj_name = obj_match.group(1)
                obj_content = obj_match.group(2)
                obj_dict = {k: v for k, v in key_value_pattern.findall(obj_content)}
                parsed[obj_name] = obj_dict
            return parsed
    
        metadata = {}
        # Recursively parse group blocks
        for match in group_pattern.finditer(text):
            group_name = match.group(1)
            group_content = match.group(2)
            metadata[group_name] = parse_block(group_content)
        return metadata
    
    def parse_hdfeos_metadata(string):
      out = _collections.OrderedDict()
      lines = [i.replace('\t','') for i in string.split('\n')]
      i = -1
      while i<(len(lines))-1:
          i+=1
          line = lines[i]
          if "=" in line:
              key,value = line.split('=', maxsplit = 1)
              if key in ['GROUP','OBJECT']:
                  endIdx = lines.index('END_{}={}'.format(key,value))
                  out[value] = parse_hdfeos_metadata("\n".join(lines[i+1:endIdx]))
                  i = endIdx
              else:
                  if ('END_GROUP' not in key) and ('END_OBJECT' not in key):
                       try:
                           out[key] = eval(value)
                       except NameError:
                           out[key] = str(value)
                       except:
                           out[key] = str(value)
      return out
    def construct_coords(ds,grid='GRID_1'):
        attrs = ds.attributes()
        metadata = parse_hdfeos_metadata(attrs['StructMetadata.0'])
        gridInfo = metadata['GridStructure'][grid]
    
    #    gridName = gridInfo['GridName']
    
        x1,y1 = gridInfo['UpperLeftPointMtrs']
        x2,y2 = gridInfo['LowerRightMtrs']
        yRes = (y1-y2)/gridInfo['YDim']
        xRes = (x1-x2)/gridInfo['XDim']
    
        #setting up coordinate grids along x and y axis
        x = _np.arange(x2,x1,xRes)
        y = _np.arange(y2,y1,yRes)[::-1]
        #set up 2D grid for plotting
        xx,yy = _np.meshgrid(x,y)
        #get projection information
        if 'soid' in gridInfo['Projection'].lower():
            pp = 'sinu'
        else:
            pp = gridInfo['Projection'].lower()
    
        #formating projection name from metadata to pyproj for sinusoidal projection
        projStr = "+proj={} +lon_0=0 +x_0=0 +y_0=0 +a={} +units=m +no_defs".format(pp,gridInfo['ProjParams'][0])
        # print(projStr)
        proj = _pyproj.Proj(projStr)
        gcs = proj.to_latlong()
        
        # Initialize Transformer object
        transformer = _pyproj.Transformer.from_proj(proj, gcs)
    
        # Convert between sinusoidal projection to lat/lon coord projection
        lon, lat = transformer.transform(xx, yy)
    
        return lon,lat

    # open file
    if isinstance(hdf, str):
        hdf=_pyhdf.SD.SD(hdf)
    else:
        pass
    attrs = hdf.attributes()
    metadata = parse_hdfeos_metadata(hdf.attributes()['StructMetadata.0'])
    
    
    orbit_times = hdf.attributes()['Orbit_time_stamp'].split()
    satellite = [o[-1] for o in orbit_times]
    orbit_times = [_pd.to_datetime(o[:-1] , format = '%Y%j%H%M') for o in orbit_times]
    
    longitude,latitude = construct_coords(hdf)
    
    #Selecte values from file for plot
    datasets = hdf.datasets()
    variables_grid5km = [name for name, info in datasets.items() if 'grid5km' in info[0][0]]
    variables_grid1km = [name for name, info in datasets.items() if 'grid1km' in info[0][0]]
    
    ds = _xr.Dataset()
    for SDS_NAME in variables_grid1km:
        sds = hdf.select(SDS_NAME) 
        data=sds.get()[:,:, ::-1]
        
        attrs = sds.attributes()
        
        if SDS_NAME != 'AOD_QA':
            data = _np.ma.masked_where(data == attrs['_FillValue'], data) #Need to remove fill values <0 because they will affect taking mean of orbits
        
        if 'scale_factor' in attrs:
            data = data * attrs['scale_factor'] #Scale factor to correct SDS values
            attrs['valid_range'] = [float(i) for i in _np.array(attrs['valid_range']) * attrs['scale_factor']]
            attrs.pop('scale_factor') # not needed anymore
        
        da = _xr.DataArray(
                        data=data,
                        dims=['datetime', "y", "x", ],
                        coords={
                            "lon": (["y", "x"], longitude),
                            "lat": (["y", "x"], latitude),
                            "datetime": orbit_times,
                        }
        )
        
        # attach atributes
        da.attrs = attrs
        ds[SDS_NAME] = da
    
    ds.datetime.attrs['long_name'] = 'Orbit_time_stamp'
    
    # qc assesment based on aod qc, there is a lot more
    #### Macht nicht wirklich sinn!!! There is only good data... moep
    
    qa = _xr.full_like(ds.AOD_QA, 3) # quality assesment
    qa.attrs = {}
    
    encoding = '016b'
    nu2bits = _np.vectorize(lambda x: format(x, encoding)[::-1][8:12])
    aod_qa_asbits = _xr.apply_ufunc(nu2bits, ds.AOD_QA)
    
    # Good Quality
    # ------------
    # * 0000 --- Best quality
      
    # Medium Quality
    # --------------
    # * 0011 --- There is 1 neighbor cloud
    # * 0100 --- There is >1 neighbor clouds
    # * 1011 --- Land, Research Quality: AOD retrieved but CM is possibly cloudy
    
    # Low Quality
    # -----------
    # * 1001 --- Retrieved AOD is low (<0.05) due to glint
    # * 1010 --- AOD within +-2km from the coastline is replaced by nearby AOD
    
    # Invalid
    # -------
    # * 0001 --- Water Sediments are detected (water)
    # * 0101 --- No retrieval (cloudy, or whatever)
    # * 0110 --- No retrievals near detected or previously detected snow
    # * 0111 --- Climatology AOD: altitude above 3.5km (Water), and 4.2km (Land)
    # * 1000 --- No retrieval due to sun glint (water)
    
    qa.values = _np.where(aod_qa_asbits == '0000', 0, qa)
    
    qa.values = _np.where(aod_qa_asbits == '0011', 1, qa)
    qa.values = _np.where(aod_qa_asbits == '0100', 1, qa)
    qa.values = _np.where(aod_qa_asbits == '1011', 1, qa)
    
    qa.values = _np.where(aod_qa_asbits == '1001', 4, qa)
    qa.values = _np.where(aod_qa_asbits == '1010', 2, qa)
    
    qa.values = _np.where(aod_qa_asbits == '0001', 3, qa)
    qa.values = _np.where(aod_qa_asbits == '0101', 3, qa)
    qa.values = _np.where(aod_qa_asbits == '0110', 3, qa)
    qa.values = _np.where(aod_qa_asbits == '0111', 3, qa)
    qa.values = _np.where(aod_qa_asbits == '1000', 3, qa)
    
    qa.values = _np.where(ds.AOD_QA == 0, 3, qa)
    ds['AOD_qa_assest'] = qa
    
    ds['DQF'] = ds.AOD_qa_assest
    ########################
    # attach some attributes
    parsed_data = parse_metadata(hdf.attributes()[list(hdf.attributes().keys())[4]])
    
    version = parsed_data['INVENTORYMETADATA']['PGEVERSION']['VALUE']
    ds.attrs['version'] = version
    ds.attrs['dataset_name'] = parsed_data['INVENTORYMETADATA']['LOCALGRANULEID']['VALUE'].split('.')[0]
    si = EOS_AOD(ds, product_version=ds.attrs['version'])
    return si #ds#, hdf, metadata



class EOS_AOD(_pylab.GeosSatteliteProducts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.valid_qf = [0,1]
        
        if self.product_info['version'] in ['6.1.19', '6.1.20','6.1.21','6.1.22','6.1.23','6.1.24','6.1.25',]:
            
            global_qf = [{'high':   [0], 
                          # 'medium': [1],
                          # 'low':    [2],
                          'bad':    [3]}]
            self.qf_managment = _pylab.QfManagment(self, 
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
            raise _pylab.GoesExceptionVerionNotRecognized(message = f"Version {self.product_info['version']} not recognized.")