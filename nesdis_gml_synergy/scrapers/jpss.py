#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:39:20 2023

@author: hagen

Known issues: 
    mapping edge issue in longiturdinal direction:
        If a site is very close to the edge with respect to its longitude 
        direction, the area could touch the edge of the scann area and create a 
        bias. The same is not happening for the Latitude as the cloud_interface 
        takes the two closest scannes in that direction.
"""

import nesdis_gml_synergy.satlab as ngs
# import nesdis_gml_synergy.info as ngsinf
import nesdis_gml_synergy.cloud_interface as ngsci
# import nesdis_aws

import multiprocessing
import time
import psutil

# import atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad.surfrad as atmsrf

import numpy as _np
import pathlib as _pl
import pandas as _pd
import xarray as _xr

class JPSSSraper(object):
    def __init__(self, 
                 start = '20200822 00:00:00', end = '20200828 00:00:00', 
                 sites = {'lon': -105.2705, 'lat': 40.015, 'alt': 1500, 'abb': 'bld'},
                 product = 'AOD', satellite = 'NOAA 20', sensor = 'VIIRS', 
                 p2fld_out = '/export/htelg/tmp/', prefix = 'projected2surfrad',
                 reporter = None,
                 overwrite = False, 
                 verbose = False):
        
        self.reporter = reporter
        self.start = start
        self.end = end
        self.satellite = satellite
        self.prefix = prefix
        self.sensor = sensor
        self.product=product
        self.p2fld_out = _pl.Path(p2fld_out)
        self.sites = sites
        # self.sites = atmsrf.network.stations.list[:3]
        self.overwrite = overwrite
        self.verbose = verbose
        
        self._workplan = None
        
        # for automation
        self.no_processed_success = self.no_processed_error = self.no_processed_warning = 0
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            #Make the workplan
            dates = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq='d', inclusive = 'left'), columns = ['site',])
            
            for e,site in enumerate(self.sites):
                dt = dates.copy()
                dt['p2f_out'] = dt.apply(lambda row: self.p2fld_out.joinpath(f"{self.prefix}_{self.satellite.replace(' ','')}_{self.sensor}_{self.product}_{site.abb}_{row.name.year:04d}{row.name.month:02d}{row.name.day:02d}.nc"), axis = 1).values
                dt['site'] = site.abb
                if e == 0:
                    workplan = dt
                else:
                    workplan = _pd.concat([workplan, dt])

            if not self.overwrite:
                workplan = workplan[~(workplan.apply(lambda row: row.p2f_out.is_file(), axis = 1))]
                
            self._workplan = workplan
        return self._workplan 
    
    def download_and_overpasses(self, date, sites):
        verbose = self.verbose
        verbose = True
        if verbose:
            print('fct call - download_and_overpasses', end = ' ... ')
            print(f'sites = {sites}')
        end = date + _pd.to_timedelta(0.9999999, 'd') # just under 1 will garantie that we stay in the same day
        # end = date + _pd.to_timedelta(1, 'd') # just under 1 will garantie that we stay in the same day
        print(date)
        print(end)
        # print(type(date))
        # print(type(end))
        query = ngsci.AwsQuery(path2folder_local='/export/htelg/tmp/',
                                    satellite=self.satellite,
                                    sensor = self.sensor,
                                    product= self.product,
                                    # scan_sector=None,
                                    # start='2022-11-13 12:00:00',
                                    # end='2022-11-14 13:00:00',
                                    start=date,
                                    end=end,
                                    site = sites
                                   )
        query.aws.clear_instance_cache()
        if query.workplan.shape[0] != 0:
            if verbose:
                print(f'Downloading {query.workplan.shape[0]} files', end = ' ... ')
            query.download()
        if verbose:
            if verbose:
                print('done!')
        return query
    
    ## work it up by site (typically there is only one overpass, non the less)
    def process_single_day(self, daygroup, error_queue = None, save = True, stop_after_first = False, remove_original_files = True):
        """
        Processes a single day (based on group_by)

        Parameters
        ----------
        daygroup : group element or int
            if you provide an integer the #ths goup will be processed.

        Returns
        -------
        None.

        """
        try:
            if isinstance(daygroup, int):
                daygroup = list(self.workplan.groupby(self.workplan.index))[daygroup] # 
    
        # if 1:
            if self.verbose:
                print('fct call - process_single_day', end = ' ... ')
            date, dgrp = daygroup
            sitesonday = [s for s in self.sites if s.abb in dgrp.site.values]
            query = self.download_and_overpasses(date, sitesonday)
            if query.overpassplan.shape[0] == 0:
                if self.verbose:
                    print('no overpass or matching file found?!?')
                # ds_out.to_netcdf(p2f_out)
                return None
            
            #### loop over site goups
            ### this can still contain multiple overpasses!! 
            for s, sgrp in query.overpassplan.groupby('site'):
                self.tp_sgrp = sgrp
                p2f_out = dgrp.where(dgrp.site == s).dropna().p2f_out[0]
            # loop over different overpasses at site; typically only one, but can be more in high latitudes
                for e,(opt, op) in enumerate(sgrp.iterrows()):
                    self.tp_op = op
                    self.tp_opt = opt
                    site = [i for i in self.sites if i.abb == s][0]
                    
                    # p2f = [sgrp.path2file_local1[0], sgrp.path2file_local2[0]]
                    p2f = [op.path2file_local1 , op.path2file_local2] 
                    # p2f = [sgrp.path2file_local1[0].as_posix(), sgrp.path2file_local2[0].as_posix()]
                    ngsinst = ngs.open_file(p2f, verbose=self.verbose)
                    
                    #remove some variables (before processing) more are removed after processing
                    for var in ['StartRow', 'StartColumn', 'MeanAOD', 'MeanAODHighQuality']:
                        ngsinst.ds = ngsinst.ds.drop_vars(var)
                        
                    projection = ngsinst.project_on_sites(site)
                    
                    # merge closest gridpoint and area
                    point = projection.projection2point.copy()#.sel(site = 'TBL')
                    
                    point['DQF'] = point.DQF.astype(int) # for some reason this was float64... because there are some nans in there
                    
                    # change var names to distinguish from area
                    if self.verbose:
                        print(f'ngsinst.valid_2D_variables: {ngsinst.valid_2D_variables}')
                    for var in ngsinst.valid_2D_variables:
                        point = point.rename({var: f'{var}_on_pixel',})
                        if f'{var}_DQF_assessed' in point.variables:
                            point = point.rename({f'{var}_DQF_assessed': f'{var}_on_pixel_DQF_assessed',})
                    point = point.rename({'DQF': 'DQF_on_pixel'})
            
                    # merge aerea and point
                    ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
                    
                    #remove some variables (after processing) more are removed berfore processing
                    for var in ['QCExtn', 'QCTest', 'QCInput', 'QCPath', 'QCRet',]:
                        ds = ds.drop_vars(var)
                    
                    # add a time stamp
                    ds = ds.expand_dims({'datetime': [opt]}, )
                            
                    # global attribute
                    ds.attrs['info'] = ('This file contains a projection of satellite data onto designated measurment sites.\n'
                                         'It includes the closest pixel data as well as the average over circular\n'
                                         'areas with various radii. Note, for the averaged data only data is\n'
                                         'considered with a qulity flag given by the prooduct class in the\n'
                                         'nesdis_gml_synergy package.')
                    
    
                    self.tp_ds = ds
                    
                    #### add awspaths
                    df = _pd.DataFrame([p.as_posix() for p in op[['path2file_aws1', 'path2file_aws1']]])
                    df.index.name = 'aws_file'
                    df.columns.name = 'datetime'
                    ds['aws_paths'] = df
                    
                    #### add observer viewing angle
                    df = _pd.DataFrame(op[['obs_azimus_angle', 'obs_elevation_angle']])
                    df.index.name = 'obs_angle'
                    df.columns.name = 'datetime'
                    df = df.rename({'obs_azimus_angle' : 'azimus', 'obs_elevation_angle' : 'elevation'})
                    ds['observer_viewing_angle'] = df
                    
                    #### add site info to attrs
                    ds.attrs['site_abbreviation'] = op.site
                    ds.attrs['site_lon'] = float(ds.lon_station)
                    ds.attrs['site_lat'] = float(ds.lat_station)
                    
                    
                    if e == 0:
                        ds_out = ds
                    else:
                        ds_out = _xr.concat([ds_out, ds],'datetime')
                        
                #### remove site dim/coord
                ### Since we separate different sites, this is not needed
                ds_out = ds_out.sel(site = ds.site.values[0])
                ds_out = ds_out.drop_vars(['site', 'lat_station', 'lon_station'])
                
                
                #### save
                if save:
                    ds_out.to_netcdf(p2f_out)
                if stop_after_first:
                    break
            # remove the satellite files to preserve storage
            if remove_original_files:
                for path in _np.unique(_pd.concat([query.overpassplan.path2file_local1,query.overpassplan.path2file_local2]).values):
                    path.unlink()
            if self.verbose:
                print('done')
        except Exception as e:
            error_queue.put(e)
            ds_out = None
        return ds_out

    def process(self, max_processes = 2, timeout = 300, sleeptime = 0.5):
        # only if spawning new processes will every thing work well
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force = True)
        
        iterator = iter(self.workplan.groupby(self.workplan.index)) # 
        process_this = self.process_single_day
        
        processes = []
        error_queue = multiprocessing.Queue()
        while 1:      
            #### catch errors
            # catch errors in the individual subprocess, to avoid scenarios where all files are downloaded but not processed
            # Also, here is the place where certain errors can be filtered out.
            if not error_queue.empty():
                e = error_queue.get()
                raise(e)
            
            #### report current progress
            if not isinstance(self.reporter, type(None)):
                self.reporter.log()
                
            for process in processes:
                # process.join(timeout=2)
                if process.is_alive():
                    p = psutil.Process(process.pid)
                    dt_in_sec = (_pd.Timestamp.now(tz = 'utc') - _pd.to_datetime(p.create_time(), unit = 's', utc = True))/ _pd.to_timedelta(1,'s')
                    assert(dt_in_sec > 0), 'process elaps time is smaller 0, its process creation time is probably not in utc! todo: find out how to determine the timezone that is used by psutil'
                    # print(dt_in_sec)
                    if dt_in_sec > timeout:
                        print(f"Process for number {process.name} exceeded the timeout and will be terminated.")
                        process.terminate()
                else:
                    proc = processes.pop(processes.index(process))
                    if not isinstance(self.reporter, type(None)):
                        if proc.exitcode == 0:
                            self.reporter.clean_increment()
                        elif proc.exitcode == -15: #process was killed due to timeout. I had that as -9 before, double check if both exit codes are possible?
                            self.reporter.errors_increment()
                        elif proc.exitcode == 1: #process generated an exception that should be rison further down
                            self.reporter.errors_increment()
                        else:
                            assert(False), f'exitcode is {proc.exitcode}. What does that mean?'
                    #### TODO: Test what the result of this process was. If it resulted in an error, make sure you know the error or stopp the entire process!!!
                print('.', end = '')
                
            if len(processes) >= max_processes:  
                time.sleep(sleeptime)
                continue
            else:
                try:
                    arg = next(iterator)
                except StopIteration:
                    # print('reached last number')
                    if len(processes) == 0:
                        break
                    else:         
                        time.sleep(sleeptime)
                        continue
                    
                process = multiprocessing.Process(target=process_this, args=(arg,error_queue), name = 'jpssscraper')
                process.daemon = True
                processes.append(process)
                process.start()
                
        #### final report
        if not isinstance(self.reporter, type(None)):
            self.reporter.log(overwrite_reporting_frequency=True)
        
        # print("All processes completed.")