#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:31:32 2025

@author: hagen
"""

import GeoLeoXtract as glx

import pathlib as pl
# import pygrib
import pandas as pd
import numpy as np
import xarray as xr
from .opt_imports import pygrib
 

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

# from hrrr_scraper import extra

# tmp = pkg_resources.path('hrrr_scraper.extra', 'hrrr_2d_grb_info_matched.xlsx')
# tmp = pd.read_excel(tmp)
# tmp = next(tmp)
# print(f'tmp: {tmp}')
# p2script = pl.Path(__file__).resolve()
# p2base = p2script.parent.parent.as_posix()

# parameters: https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml
params = [### verticle profiles
            {'parameterName' : 'Mass density', 'typeOfLevel' : 'hybrid', 'my_name': 'aerosol_mass_density_vp'},
            {'parameterName' : 'Specific humidity', 'my_name': 'specific_humidity_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : '32', 'my_name': 'fraction_of_cloud_cover_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : 'Geopotential height', 'my_name': 'level_height_geo_potential_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : 'Temperature', 'my_name': 'temperature_vp', 'typeOfLevel' : 'hybrid'},
            
            {'parameterName': 'Pressure', 'typeOfLevel': 'hybrid', 'my_name': 'pressure_vp'},
            {'name': 'Fraction of cloud cover', 'typeOfLevel': 'hybrid', 'my_name': 'fraction_of_cloud_cover_vp'},
            {'parameterName': 'Specific humidity', 'typeOfLevel': 'hybrid', 'my_name': 'specific_humidity_vp'},
            {'name': 'U component of wind', 'typeOfLevel': 'hybrid', 'my_name': 'u_component_of_wind_vp'},
            {'name': 'V component of wind', 'typeOfLevel': 'hybrid', 'my_name': 'v_component_of_wind_vp'},
            {'name': 'Vertical velocity', 'typeOfLevel': 'hybrid', 'my_name': 'vertical_velocity_vp'},
            {'name': 'Turbulent kinetic energy', 'typeOfLevel': 'hybrid', 'my_name': 'turbulent_kinetic_energy_vp'},
            
            {'parameterName': 'Cloud mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'cloud_mixing_ratio_vp'},
            {'parameterName': '82', 'typeOfLevel': 'hybrid', 'my_name': 'ice_water_mixing_ratio_vp'},
            
            # trouble getting all the work done ... exclude the next two
            {'parameterName': 'Rain mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'rain_mixing_ratio_vp'},
            {'parameterName': 'Snow mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'snow_mixing_ratio_vp'},


            ### at surface
            {'parameterName' : 'Mass density', 'typeOfLevel' : 'heightAboveGround', 'my_name': 'aerosol_mass_density_ground_level'},
            {'parameterName' : 'Downward long-wave radiation flux', 'my_name': 'downward_long_wave_radiation_flux_surface'},
            {'parameterName' : 'Upward long-wave radiation flux', 'my_name': 'upward_long_wave_radiation_flux_surface', 'typeOfLevel' : 'surface'},
            {'parameterName' : 'Downward short-wave radiation flux', 'my_name': 'downward_short_wave_radiation_flux_surface'},
            {'name' : 'Upward short-wave radiation flux', 'my_name': 'uward_short_wave_radiation_flux_surface', 'typeOfLevel' : 'surface'},
            {'name' : 'Orography', 'my_name': 'orography'},
            {'parameterName' : 'Temperature', 'my_name': 'temperature_surface', 'typeOfLevel' : 'surface'}, 
            # {'parameterName': 'Geopotential height', 'typeOfLevel':'isothermal', 'level': 253, 'my_name': 'planetary_boundary_layer_height'},  
            {'parameterName': 'Planetary boundary layer height', 'my_name': 'planetary_boundary_layer_height'},  
            
            {'parameterName': 'Visibility', 'typeOfLevel': 'surface', 'my_name': 'visibility_surface'},
            {'parameterName': 'Wind speed (gust)', 'typeOfLevel': 'surface', 'my_name': 'wind_speed_gust_surface'},
            {'name': 'Surface pressure', 'typeOfLevel': 'surface', 'my_name': 'surface_pressure_surface'},
            {'parameterName': 'Plant canopy surface water', 'typeOfLevel': 'surface', 'my_name': 'plant_canopy_surface_water_surface'},
            {'parameterName': 'Snow cover', 'typeOfLevel': 'surface', 'my_name': 'snow_cover_surface'},
            {'parameterName': 'Snow depth', 'typeOfLevel': 'surface', 'my_name': 'snow_depth_surface'},
            {'parameterName': 'Precipitation rate', 'typeOfLevel': 'surface', 'my_name': 'precipitation_rate_surface'},
            {'parameterName': 'Sensible heat net flux', 'typeOfLevel': 'surface', 'my_name': 'sensible_heat_net_flux_surface'},
            {'parameterName': 'Latent heat net flux', 'typeOfLevel': 'surface', 'my_name': 'latent_heat_net_flux_surface'},
            {'parameterName': 'Ground heat flux', 'typeOfLevel': 'surface', 'my_name': 'ground_heat_flux_surface'},
            # {'name': 'GPP coefficient from Biogenic Flux Adjustment System', 'typeOfLevel': 'surface', 'my_name': 'gpp_coefficient_from_biogenic_flux_adjustment_system_surface'},
            {'parameterName': 'Convective available potential energy', 'typeOfLevel': 'surface', 'my_name': 'convective_available_potential_energy_surface'},
            {'parameterName': 'Convective inhibition', 'typeOfLevel': 'surface', 'my_name': 'convective_inhibition_surface'},
          
            ### column
            {'parameterName' : 'Total column', 'my_name': 'aerosol_integrated_density_column'},
            {'parameterName' : 'Precipitable water', 'my_name': 'precipitable_water_column'},
            {'parameterName' : 'Total cloud cover', 'my_name': 'total_cloud_cover', 'typeOfLevel' : 'atmosphere'},
            {'parameterName' : '102', 'my_name': 'aerosol_optical_depth'},
            # clouds
            {'parameterName' : 'Geopotential height', 'my_name': 'cloud_base_gph', 'typeOfLevel' : 'cloudBase'},
            {'parameterName' : 'Geopotential height', 'my_name': 'cloud_top_gph', 'typeOfLevel' : 'cloudTop'},
            {'shortName' : 'refc', 'my_name': 'composite_reflectivity', 'typeOfLevel' : 'atmosphere'},
          
#           {'parameterName' : '', 'my_name': '', 'typeOfLevel' : ''},
         ]

attrs = [
 'parameterCategory',
 'parameterNumber',
 'parameterUnits',
 'parameterName',
 'paramIdECMF',
 'paramId',
 'shortNameECMF',
 'shortName',
 'unitsECMF',
 'units',
 'nameECMF',
 'name',
 'cfNameECMF',
 'cfName',
 'cfVarNameECMF',
 'cfVarName',
 'modelName',
]




def open_grib_file(fname, parameter_subset = None, external_params = False, grab_basics = False, 
                   raise_error_when_varible_missing = True,
                   verbose = False):
    
    if isinstance(fname, pl.Path):
        fname = fname.as_posix()
        
    with pygrib.open(fname) as grbs:
        if grab_basics:
            basics = {}
            grb = grbs[1]
            basics['forecastTime'] = grb.forecastTime
            basics['cycledatetime'] = pd.to_datetime(f'{grb.year}-{grb.month:02d}-{grb.day:02d} {grb.hour:02d}:{grb.minute:02d}:{grb.second:02d}')
            out = basics
        else:
            # grbs = pygrib.open(fname)
            ds = read_selected_fields(grbs, 
                                      parameter_subset = parameter_subset, 
                                      external_params = external_params, 
                                      # vp = vp, 
                                      raise_error_when_varible_missing = raise_error_when_varible_missing,
                                      verbose = verbose) 
            #### cycle and forcast hour
            grb = grbs[1]
            ds.attrs['forecast_time'] = f'{grb.forecastTime:02d}'
            ds.attrs['cycle_datetime'] = f'{grb.year}-{grb.month:02d}-{grb.day:02d} {grb.hour:02d}:{grb.minute:02d}:{grb.second:02d}'
            ds.attrs['title'] = 'HRRR'
            if verbose:
                print('closing grib file')
            
            #### imitate a QC-flag so the QC manager is not complaining, not sure if this is still needed?
            ds['DQF'] = xr.zeros_like(ds[max(ds.data_vars, key=lambda var: len(ds[var].dims))])

            # out = HrrrWrfNat(ds)
            out = glx.satlab.HRRR(ds, 
                product_version = 0, #can find what HRRR version is actually producing the grib file
               )
            # out = ds
    return out


def read_selected_fields(grbs, vp = True, parameter_subset = None, external_params = False,  raise_error_when_varible_missing = True, verbose = True):
    if verbose:
        print('function -> read_selected_fields')
        
    def grb_to_grid(grb_obj):
        """Takes a single grb object containing multiple
        levels. Assumes same time, pressure levels. Compiles to a cube"""
        n_levels = len(grb_obj)
        levels = np.array([grb_element['level'] for grb_element in grb_obj])
        # print(f'{levels.dtype}, {levels.max()}')
        indexes = np.argsort(levels)#[::-1] # highest pressure first
        cube = np.zeros([n_levels, grb_obj[0].values.shape[0], grb_obj[1].values.shape[1]], dtype = np.float32)
        for i in range(n_levels):
            cube[i,:,:] = grb_obj[indexes[i]].values
            
        # print(f'{cube.dtype}, {cube.max()}')
        cube_dict = {'data' : cube, 'units' : grb_obj[0]['units'],
                     'levels' : levels[indexes]}
        return cube_dict
    
    if isinstance(parameter_subset, type(None)):
        parameter_dict_list = params
    else:
        parameter_dict_list = [par for par in  params if par['my_name'] in parameter_subset ]
        
        assert(len(parameter_subset) == len(parameter_dict_list)), 'not all param selection have been found in the parameter dictionary, there has to be a spelling error'
        
    
    grb = grbs[1] # just a random parameter ... 76 is an the ground smoke concentration ... 76 did not work with custom files
    # lat, lon = grb.latlons()
    lat, lon = [i.astype(np.float32) for i in  grb.latlons()]

    ds = xr.Dataset()

#### TODO: extend to other file formats
#### TODO: extend to 3d
    if external_params:
    #### 3d parameters
        #read variable match table
        
        if isinstance(external_params, dict):
            file_type = external_params['file_type']
            variables = external_params['variables']
        
        else:
            file_type = external_params
            variables = 'all'
        
        if file_type == 'HRRRv4_2d':
            # p2varmatch = './extra/hrrr_2d_grb_info_matched.xlsx'
            # df = pd.read_excel(p2varmatch)
            df = pd.read_excel(pkg_resources.open_binary('hrrr_scraper.extra', 'hrrr_2d_grb_info_matched.xlsx'))
        else:
            assert(False), f'Currently external_params can only have the file_type "HRRRv4_2d". {file_type} not recognized.'
        ## pick only those with netcdf names
        df = df[~(df['netcdf variable name'].isna())]
        
        # get the ones that are in the selected variables
        if variables != 'all':
            df = df[df['netcdf variable name'].isin(variables)]
        # params = []
        for idx, row in df.iterrows():
            if row.typeOfLevel =='hybrid':
                assert(False), 'not working yet, program dude!'  
                
        #### get the 2d stuff
        param_sel = []
        for idx, row in df.iterrows():
            if row.typeOfLevel =='hybrid':
                continue
            var_name = row.pop('netcdf variable name')
            long_name = row.pop('long name')
            units_comments = row.pop('units in extracted/other comments')
            grib2_variable_name = row.pop('grib2 variable name')
            msg = row.pop('messagenumber')
                        
            try:
                grbsel = grbs.select(**row.to_dict())
            except ValueError as err:
                if err.args[0] == "no matches found":
                    print('2d problem:', end = ' ')
                    print(row)
                raise
            # return grbs, row
                
                
            # else:
            #     try:
            #         grbsel = grbs.select(**part)
            #     except ValueError:
            #         print('2d problem:', end = ' ')
            #         print(part)
            #         continue
    
            assert(len(grbsel) == 1), f'The grib select function retured {len(grbsel)} messages instead of 1.'
            grb = grbsel[0]
            assert(grb.messagenumber == msg), f'Message number is {grb.messagenumber}. Config file suggests {msg}.'
            
            # da = xr.DataArray(grb.values.astype(np.float32),coords = {'lat':(['x','y'], lat),
            #                                    'lon':(['x','y'], lon)}, 
            #              dims = ['x','y'])
            da = xr.DataArray(grb.values.astype(np.float32),coords = {'lat':(['y','x'], lat),
                                               'lon':(['y','x'], lon)}, 
                         dims = ['y','x'])
            
            da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
            da.attrs['long_name'] = long_name
            da.attrs['comments'] = units_comments
            da.attrs['grib2 variable name'] = grib2_variable_name
            ds[var_name] = da
        
    #### 3d parameters
    ### get all parameters with the hybrid typeOfLevel
    else:
        if vp:
            if verbose:
                print('reading 3D parameters')
                
            param_sel = []
            for par in parameter_dict_list:
                if 'typeOfLevel' in par.keys():
                    if par['typeOfLevel'] == 'hybrid':
                        param_sel.append(par)
        
            for par in param_sel:
                # print(par)
                part = par.copy()
                part.pop('my_name')
                try:
                    grbsel = grbs.select(**part)
                except ValueError:# as err:
                    # if err.args[0] == "no matches found":
                        # print(err, end = ' ')
                    print('problem', end = ': ')
                    print(part)
                    continue
                        
                    # raise
                # grbsel
        
                out = grb_to_grid(grbsel)
        
                levels = out['levels']
        
                da = xr.DataArray(out['data'],coords = {'lat':(['y','x'], lat),
                                                   'lon':(['y','x'], lon),
                                                   'level' :levels}, 
                             dims = ['level','y','x'])
                grb = grbsel[0]
                da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
                ds[par['my_name']] = da
        
        # print('3d done')
        ### get the 2d stuff
        if verbose:
            print('get 2D params')
    
        param_sel = []
        for par in parameter_dict_list:
            if 'typeOfLevel' in par.keys():
                if par['typeOfLevel'] == 'hybrid':
                    continue
            param_sel.append(par)
    
        for par in param_sel:
            part = par.copy()
            part.pop('my_name')
                
            # grbsel = grbs.select(**part)
            if verbose:
                print(f'get: {part}', end = ' ... ')
            try:
                grbsel = grbs.select(**part)
            except ValueError as err:
                if err.args[0] == "no matches found":
                    print('2d problem:', end = ' ')
                    print(part, end = '')
                    if not raise_error_when_varible_missing:
                        print(' ... skip')
                        continue
                    else:        
                        raise
                else:
                    raise
                
                
            # else:
            #     try:
            #         grbsel = grbs.select(**part)
            #     except ValueError:
            #         print('2d problem:', end = ' ')
            #         print(part, end = '... skip\n')
            #         continue
            
            if verbose:
                print('create Dataarray and add to dataset', end = ' ... ')
            assert(len(grbsel) == 1)
            grb = grbsel[0]
            da = xr.DataArray(grb.values.astype(np.float32),coords = {'lat':(['y','x'], lat),
                                               'lon':(['y','x'], lon)}, 
                         dims = ['y','x'])
            da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
            ds[par['my_name']] = da
            
            if verbose:
                print('done')
    
    
    return ds