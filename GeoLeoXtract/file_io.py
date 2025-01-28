#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:31:44 2025

@author: hagen
"""

import xarray as _xr
import magic as _magic
import pathlib as _pl
import pyhdf as _pyhdf
import collections as _collections
import numpy as _np
import re as  _re
import pyproj as _pyproj
import pandas as _pd


import gc


from . import nasa_tempo
from . import satlab



def open_file(p2f, auto_assign_product = True, bypass_time_unit_error = True, extent = None ,verbose = False):
    """
    Open a satellite data file. Probably only works for GOES

    Parameters
    ----------
    p2f : string or xarray.Dataset
        Path to file or a xarray.Dataset.
    bypass_time_unit_error: bool, optional.
        In the past some files have an error in the time variable. This allows 
        you to open it anyway. Also, there are other ways to get the time!
    extend : list, optional
        Select a particular area by longitude (lon) and latitude (lat): 
            [min(lon), max(lon), min(lat), max(lat)]. The default is None.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    classinst : TYPE
        DESCRIPTION.

    """
    if isinstance(p2f, _xr.Dataset):
        ds = p2f
    else:
        try:
            if isinstance(p2f, list):
                dslist = []
                for fn in p2f:
                    # currently this is only used for leo products, will probably cause errors when trying to use for something else
                    ftype = _magic.detect_from_filename(fn).name
                    assert(ftype == 'Hierarchical Data Format (version 5) data'), 'Only "Hierarchical Data Format (version 5) data" can be read from a list of files. if you want this to work for other file formats (e.g. HDFv4) ... fix this'
                    
                    dst = _xr.open_dataset(fn)
                    dst = dst.where(~dst.Latitude.isnull(), drop = True)
                    dst = dst.where(~dst.Longitude.isnull(), drop = True)
                    dslist.append(dst)
                ds = _xr.concat(dslist, dim = 'Rows')
            else:
                ftype = _magic.detect_from_filename(p2f).name
                if ftype == 'Hierarchical Data Format (version 5) data':
                    with _xr.open_dataset(p2f) as ds:
                        project = None
                        if 'project' in ds.attrs:
                            project = ds.attrs['project'].lower()
                    if project == 'tempo':
                        si = nasa_tempo.open(p2f)
                        if verbose:
                            print('NASA Tempo file detected')
                        return si
                    else:
                        ds = _xr.open_dataset(p2f)
            
                elif ftype == 'Hierarchical Data Format (version 4) data':
                    if isinstance(p2f, _pl.Path):
                        p2f = p2f.as_posix()
                    
                    hdf=_pyhdf.SD.SD(p2f)
                    if hdf.attributes()['identifier_product_doi'] == '10.5067/MODIS/MCD19A2.006':
                        if verbose:
                            print(f'detected "10.5067/MODIS/MCD19A2.006" product')

                        ds = read_Modis_MCD19A2(hdf)
                        si = satlab.EOS_AOD(ds, product_version=ds.attrs['version'])
                        
                        hdf.end()
                        del hdf
                        gc.collect()
                        
                        return si

                p2f = [_pl.Path(p2f),]
        except ValueError as err:
            if not bypass_time_unit_error:
                raise
            else:
                if 'unable to decode time units' in err.args[0]:
                    ds = _xr.open_dataset(p2f,decode_times=False,)
                else:
                    raise
    
    if 'dataset_name' in ds.attrs.keys():    
        product_name = ds.attrs['dataset_name'].split('_')[1]
    elif 'title'in [k.lower() for k in ds.attrs.keys()]:   
        if 'Title' in ds.attrs.keys():
            ds.attrs['title'] = ds.attrs.pop('Title')
        # e.g the experimental Surface radiation budget product and NOAA20 products did not have data_set attribute
        product_name = ds.attrs['title']
    else:
        assert(False), 'NetCDF file has no attribute named "dataset_name", or "title"'
            
        
    if verbose:
        print(f'product name: {product_name}')
    # if product_name == 'ABI-L2-AODC-M6':
    #     classinst = ABI_L2_AODC_M6(ds)
    if not auto_assign_product:
        classinst = satlab.GeosSatteliteProducts(ds)
        return classinst

    #### VIRRS products
    if product_name in ['AEROSOL_AOD_EN', 'JRR-AOD']:
        pv = _np.unique([float(p.name.split('_')[1][slice(1,4,2)])/10 for p in p2f])
        assert(len(pv) == 1), f'version of files is different ({pv})'
        pv = pv[0]
        if verbose:
            print(f'Found AEROSOL_AOD_EN version {pv}')
        classinst = satlab.JRR_AOD(ds, product_version = pv)

    #### ABI products    
    elif 'ABI-L2-AODC' in product_name:
        classinst = satlab.ABI_L2_AOD(ds)
    elif product_name[:-1] == 'ABI-L2-MCMIPC-M':
        classinst = satlab.ABI_L2_MCMIPC_M6(ds)
    elif product_name[:-4] == 'ABI-L2-LST':
        classinst = satlab.ABI_L2_LST(ds)
        if verbose:
            print('identified as: ABI-L2-LSTC-M6')
    elif product_name[:-4] == 'ABI-L2-COD':
        classinst = satlab.ABI_L2_COD(ds)
        if verbose:
            print('identified as: ABI_L2_COD.')
    elif product_name[:-4] == 'ABI-L2-ACM':
        classinst = satlab.ABI_L2_ACM(ds)
        if verbose:
            print('identified as: ABI_L2_ACM.')
    elif product_name[:-4] == 'ABI-L2-ADP':
        classinst = satlab.ABI_L2_ADP(ds)
        if verbose:
            print('identified as: ABI_L2_ADP.')
    elif product_name[:-4] == 'ABI-L2-ACHA':
        classinst = satlab.ABI_L2_ACHA(ds)
        if verbose:
            print('identified as: ABI_L2_ACHA.')
    elif product_name[:-4] == 'ABI-L2-CTP':
        classinst = satlab.ABI_L2_CTP(ds)
        if verbose:
            print('identified as: ABI_L2_CTP.')
    elif product_name[:-4] == 'ABI-L2-DSR':
        classinst = satlab.ABI_L2_DSR(ds)
        if verbose:
            print('identified as: ABI_L2_DSR.')
    elif product_name == 'ABI L2 Shortwave Radiation Budget (SRB)':
        classinst = satlab.ABI_L2_SRB(ds)
        if verbose:
            print('identified as: ABI_L2_DSR.')
        
    else:
        classinst = satlab.GeosSatteliteProducts(ds)
        if verbose:
            print('not identified')
        # assert(False), f'The product {product_name} is not known yet, programming required.'
    
    if not isinstance(extent, type(None)):
        classinst  = classinst.select_area(extent)
        
    
    return classinst

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
    return ds#, hdf, metadata