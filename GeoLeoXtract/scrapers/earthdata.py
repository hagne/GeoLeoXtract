#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:39:20 2023

@author: hagen

Known issues: 
    mapping edge issue in longiturdinal direction:
        If a site is very close to the edge, the area could touch the edge of the scann area and create a 
        bias.
"""

import GeoLeoXtract.satlab as ngs
# import nesdis_gml_synergy.info as ngsinf
import GeoLeoXtract.cloud_interface as ngsci
# import nesdis_aws
import GeoLeoXtract as glx
import warnings as _warnings
import multiprocessing
import time
import psutil

# import atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad.surfrad as atmsrf

import numpy as _np
import pathlib as _pl
import pandas as _pd
import xarray as _xr

import requests

class GranuleMissmatchError(Exception):
    """Exception raised when a granuel was found that is not the one that is designated for that site."""
    
    def __init__(self, found_granules='NA', excepted_granules = 'NA'):
        self.message = f'Granules of found files ({found_granules}) do not match the id of the site granuele ({excepted_granules})'
        super().__init__(self.message)

class NoGranuleFoundError(Exception):
    """Exception raised when no granuel was found."""   
    def __init__(self,*args):
        self.message = 'No granule was found for this site and day.'
        super().__init__(self.message)

class MultipleFileOnServerError(Exception):
    """Exception raised when There are multiple files found on the server for the same day. They typically have the very same data in them, but are updated at a diffrerent time. If this error is muted the newer of the files will be used"""   
    def __init__(self,*args):
        self.message = 'Multiple files found on the server for the same day. They typically have the very same data in them, but are updated at a diffrerent time. If this error is muted the newer of the files will be used'
        super().__init__(self.message)

def search_granules(endpoint = 'https://cmr.earthdata.nasa.gov/search/granules.json',
                    collection_concept_id = 'C2324689816-LPCLOUD',
                    temporal =  '2022-06-02T00:00:00Z,2022-06-02T23:59:59Z',
                    point =  '-105.2705,40.015'):


    params = {
        'collection_concept_id': collection_concept_id, # version_id"061"
        # 'collection_concept_id': 'C2763289461-LPCLOUD', #version_id"006"
        # 'temporal': temporal, #'2022-06-02T00:00:00Z,2022-06-02T23:59:59Z',  # Specific date
        # 'temporal': '2019-06-02T00:00:00Z,2019-12-02T23:59:59Z',  # Specific date
        'temporal': temporal,  # Specific date
        # 'bounding_box': bbox,  # Global, adjust if necessary
        'page_size': 10  ,# Number of results to return
        # 'day_night_flag': 'both',
        'point': point,
    }
    
    response = requests.get(endpoint, params=params,
                           )
    data = response.json()
    return data


# overriding requests.Session.rebuild_auth to mantain headers when redirected
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

   # Overrides from the library to keep headers when redirected to or from
   # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

def download_url(url, path2save):
    session = SessionWithHeaderRedirection(glx.config['earthdata_credentials']['username'],
                                           glx.config['earthdata_credentials']['password'])
    
    # submit the request using the session
    response = session.get(url, stream=True)
    # print(f'response.status_code: {response.status_code}')
    
    # raise an exception in case of http errors
    response.raise_for_status()  
    
    # save the file
    with open(path2save, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=1024*1024):
            fd.write(chunk)
    return


class CMRSraper(object):
    """designed to scrape from earthdatas cmr interface"""
    def __init__(self, 
                 start = '20200822 00:00:00', end = '20200828 00:00:00', 
                 sites = {'lon': -105.2705, 'lat': 40.015, 'alt': 1500, 'abb': 'bld'},
                 product = 'AOD', satellite = 'TerraAqua', sensor = 'MODIS', 
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


        if not isinstance(sites, list):
            sites = [sites,]
        if isinstance(sites[0], dict):
            sites = [type('observatory', (), s) for s in sites]
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
    
    @workplan.setter
    def workplan(self,value):
        self._workplan = value
        return
    
    def process_single_day(self, daygroup, 
                           error_queue = None, 
                           save = True,
                           stop_after_first = False,
                           remove_original_files = True, 
                           surpress_warnings = True,
                           verbose = False,
                            skip_granule_missmatch_error = False,
                            skip_no_granule_found_error = False,
                           skip_http_error = False,
                           skip_multiple_file_on_server_error = False
                          ):
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
        ds = None
        if surpress_warnings:        
            _warnings.filterwarnings('ignore')
        
        files2remove = []
        try:
            if isinstance(daygroup, int):
                daygroup = list(self.workplan.groupby(self.workplan.index))[daygroup] # 
            # return daygroup
        # if 1:
            if verbose:
                print('fct call - process_single_day',
                      # end = ' ... '
                     )
            date, dgrp = daygroup
            for sa, sdf in dgrp.groupby('site'):
                assert(sdf.shape[0] == 1), 'This should really not be possible [id = 23113122]'
                # return sdf
                # search for granule
                site = [s for s in self.sites if sa == s.abb][0]
                start = sdf.index[0]
                self.tp_dt_start = start
                end = start + _pd.to_timedelta('23:59:59')
                temporal = f'{start.to_datetime64().astype("datetime64[s]")}Z,{end.to_datetime64().astype("datetime64[s]")}Z'    
                point = f'{site.lon},{site.lat}'    
                if verbose:
                    print('search for granule')
                data = glx.scrapers.earthdata.search_granules(temporal = temporal, point = point)
                # return data
                assert('errors' not in data), f'Errors encountered in the granula search:\n{data["errors"]}'
            
                # make download dataframe
                entries = data['feed']['entry']
                self.tp_data = data
                if len(entries) == 0:
                    if skip_no_granule_found_error:
                        print('NGFE', end = ' ')
                        continue
                    else:
                        raise NoGranuleFoundError()
                df = _pd.DataFrame(entries)
                df = df.loc[:,['title', 'updated', 'links']]
                df['name'] = df.apply(lambda row: '.'.join(row['title'].split('.')[:-1]), axis = 1)
                
                
                # assert(_np.all(df.groupby('name').count() == 1)), 'There are multiple files for the same file name (title without generation date). This happens when there are different versions with different updata dates, handle is when it accurse.'
                


                
                df['url_download'] = df.apply(lambda row: row.links[0]['href'], axis =1)
                df['p2out'] = df.apply(lambda row: _pl.Path(f'~/tmp/{row.url_download.split("/")[-1]}').expanduser(), axis = 1)
                df['granule'] = df.apply(lambda row: row.title.split('.')[2], axis = 1)
                granules = df.granule.copy()
                # return df, site
                self.tp_df_pgc = df.copy()
                self.tp_site = site



                
                # print('test baadslkdejs')
                # print(sdf)
                # print(sdf.index)
                # print(df)
                # assert(False), 'haaaaalt'




                
                df = df.where(df.granule == site.earthdata_granule).dropna()
                self.tp_df_agc = df.copy()
                
                # assert(df.shape[0] != 0), f'Granules of found files ({granules.values}) do not match the id of the site granuele ({site.earthdata_granule})'
                if df.shape[0] == 0:
                    if skip_granule_missmatch_error:
                        print('GME', end = ' ')
                        continue
                    else:
                        raise GranuleMissmatchError(granules.values, site.earthdata_granule)

                if df.shape[0] > 1: #_np.all(df.groupby('name').count() > 1):
                    self.tp_df_multifiletest = df.copy()


                    
                    # print('test baadslkdejs')
                    # print(df)
                    # print(_np.all(df.groupby('name')))
                    
                    assert(_np.all(df.groupby('name').count() > 1)),'if there are mulitple files left, they at least should have the same name but differetn update dates. So this error is unexpected.'

                    if skip_multiple_file_on_server_error:
                        print('MFOSE', end = ' ')
                        df= df.sort_values('updated', ascending = False).iloc[[0,]]
                    else:
                        raise MultipleFileOnServerError()
                assert(df.shape[0] == 1), 'There really should only be one file left at this stage.'
                download_wp = df
                self.tp_download_wp = download_wp.copy()
                if verbose:
                    print(f'download_wp: {download_wp}')
                    
                # for idx, row in download_wp.iterrows():
                row = download_wp.iloc[0]
                url = row.url_download
                path2save = row.p2out
                if path2save.is_file():
                    if verbose:
                        print('file exists, skip download')
                else:
                    if verbose:
                        print(f'downloading {url}\nto{path2save}')
                    try:
                        glx.scrapers.earthdata.download_url(url, path2save)
                    except requests.HTTPError as e:
                        if skip_http_error:
                            print('HTTPE', end = ' ')
                            continue
                        else:
                            raise

                # return None
                # Open the file and project
                ngsinst = ngs.open_file(path2save, verbose=verbose)
                
                #remove some variables (before processing) more are removed after processing
                # for var in ['StartRow', 'StartColumn', 'MeanAOD', 'MeanAODHighQuality']:
                #     ngsinst.ds = ngsinst.ds.drop_vars(var)
                    
                projection = ngsinst.project_on_sites(site)
                
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
                
                # merge aerea and point
                ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
                
                #remove some variables (after processing) more are removed berfore processing
                # for var in ['QCExtn', 'QCTest', 'QCInput', 'QCPath', 'QCRet',]:
                #     ds = ds.drop_vars(var)
                
                # add a time stamp
                # ds = ds.expand_dims({'datetime': [opt]}, )
                        
                # global attribute
                ds.attrs['info'] = ('This file contains a projection of satellite data onto designated measurment sites.\n'
                                     'It includes the closest pixel data as well as the average over circular\n'
                                     'areas with various radii. Note, for the averaged data only data is\n'
                                     'considered with a qulity flag given by the prooduct class in the\n'
                                     'GeoLeoXtract package.')
                
                
                self.tp_ds = ds
                
                #### add awspaths
                
                #### add site info to attrs
                ds.attrs['source'] = url
                ds.attrs['site_abbreviation'] = site.abb
                ds.attrs['site_lon'] = float(ds.lon_station)
                ds.attrs['site_lat'] = float(ds.lat_station)
                ds.attrs['date processed'] = f'{_pd.Timestamp.now()}'

                if save:
                    ds.to_netcdf(sdf.iloc[0].p2f_out)
                #####
                files2remove.append(path2save)

            # remove the satellite files to preserve storage
            if remove_original_files:
                if verbose:
                    print('The following files will be removed:')
                    for p2f in files2remove:
                        print(f'\t{p2f}')
                for p2f in files2remove:
                    p2f.unlink()
                    
            if self.verbose:
                print('done')
        except Exception as e:
            if verbose:
                print(e)
            if isinstance(error_queue, type(None)):
                raise 
            error_queue.put(e)
        return ds

    def process(self, max_processes = 2, timeout = 300, sleeptime = 1, 
                skip_granule_missmatch_error = False,
                skip_no_granule_found_error = False,
                skip_http_error = False,
                skip_multiple_file_on_server_error = False):          
        
        iterator = iter(self.workplan.groupby(self.workplan.index)) # 
        process_this = self.process_single_day           
        
        # only if spawning new processes will every thing work well
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force = True)
            
        processes = []
        error_queue = multiprocessing.Queue()
        while 1:      
            #### catch errors
            # catch errors in the individual subprocess, to avoid scenarios where all files are downloaded but not processed
            # Also, here is the place where certain errors can be filtered out.
            while not error_queue.empty():
                e = error_queue.get()
                do_raise = True
                msg = False
                if isinstance(e, GranuleMissmatchError):
                    if skip_granule_missmatch_error:
                        do_raise = False
                        msg = 'GME'
                elif isinstance(e, NoGranuleFoundError):
                    if skip_granule_missmatch_error:
                        do_raise = False
                        msg = 'NGFE'
                        
                if do_raise:
                    for process in processes:
                        process.terminate()
                    raise(e)
                else:
                    if msg:
                        print(msg, end = ' ')
            
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
                        elif proc.exitcode == -15: #process was killed due to timeout. 
                            self.reporter.errors_increment()
                        elif proc.exitcode == -9: #process was killed externally, e.g. by the out of memory killer, or someone killed it by hand? 
                            self.reporter.errors_increment()
                        elif proc.exitcode == 1: #process generated an exception that should be rison further down
                            self.reporter.errors_increment()
                        else:
                            if not error_queue.empty():
                                e = error_queue.get()
                                raise(e)
                            assert(False), f'exitcode is {proc.exitcode}. What does that mean?'
                    #### TODO: Test what the result of this process was. If it resulted in an error, make sure you know the error or stopp the entire process!!!
                
                
            if len(processes) >= max_processes:  
                # print('|', end = '')
                time.sleep(sleeptime)
                continue
            else:
                try:
                    arg = next(iterator)
                    # print(arg)
                except StopIteration:
                    # print('reached last number')
                    if len(processes) == 0:
                        break
                    else:         
                        time.sleep(sleeptime)
                        continue
                    
                process = multiprocessing.Process(target=process_this, 
                                                  args=(arg,error_queue),  # positional arguments
                                                  kwargs={'skip_granule_missmatch_error': skip_granule_missmatch_error,
                                                          'skip_no_granule_found_error': skip_no_granule_found_error,
                                                          'skip_http_error': skip_http_error,
                                                          'skip_multiple_file_on_server_error': skip_multiple_file_on_server_error,
                                                         },  # keyword arguments 
                                                  name = 'jpssscraper')
                process.daemon = True
                processes.append(process)
                process.start()
                print('.', end = '')
                
        #### final report
        if not isinstance(self.reporter, type(None)):
            self.reporter.log(overwrite_reporting_frequency=True)
        
        # print("All processes completed.")