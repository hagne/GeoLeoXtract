#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:37:41 2021

@author: hagen
"""
import pandas as _pd
import pathlib as _pl
import ftplib as _ftplib
import GeoLeoXtract.satlab as _satlab
import numba as _numba
import numpy as _np
import xarray as _xr
import multiprocessing as _mp
from functools import partial as _partial



@_numba.jit(nopython=True)
def get_closest_gridpoint(lon_lat_grid, lon_lat_sites):
    """using numba only saved me 5% of the time"""
#     out = {}
    lon_g, lat_g = lon_lat_grid
    # armins columns: argmin_x, argmin_y, lon_g, lat_g, lon_s, lat_s, dist_min
    out = _np.zeros((lon_lat_sites.shape[0], 7))
    
#     if len(lon_g.shape) == 3:
#         lon_g = lon_g[0,:,:]
#         lat_g = lat_g[0,:,:]
    for e,site in enumerate(lon_lat_sites):
        
        lon_s, lat_s = site
    #     lon_s = site.lon
    #     lat_s = site.lat

        p = _np.pi / 180
        a = 0.5 - _np.cos((lat_s-lat_g)*p)/2 + _np.cos(lat_g*p) * _np.cos(lat_s*p) * (1-_np.cos((lon_s-lon_g)*p))/2
        dist = 12742 * _np.arcsin(_np.sqrt(a))

        # get closest
        argmin = dist.argmin()//dist.shape[1], dist.argmin()%dist.shape[1]
        out[e,:2] = argmin        
        out[e,2] = lon_g[argmin]
        out[e,3] = lat_g[argmin]
        out[e,4] = lon_s
        out[e,5] = lat_s
        out[e,6] = dist[argmin]
    return out

@_numba.jit(nopython=True)
def get_distance(lon_lat_grid, lon_lat_sites):
    """using numba only saved me 5% of the time"""
#     out = {}
    lon_g, lat_g = lon_lat_grid
    # armins columns: argmin_x, argmin_y, lon_g, lat_g, lon_s, lat_s, dist_min
    # out = _np.zeros((lon_lat_sites.shape[0], 7))
    dist_ar = _np.zeros((lon_lat_sites.shape[0],) + lon_lat_grid[0].shape)
    
#     if len(lon_g.shape) == 3:
#         lon_g = lon_g[0,:,:]
#         lat_g = lat_g[0,:,:]
    for e,site in enumerate(lon_lat_sites):
        
        lon_s, lat_s = site
    #     lon_s = site.lon
    #     lat_s = site.lat

        p = _np.pi / 180
        a = 0.5 - _np.cos((lat_s-lat_g)*p)/2 + _np.cos(lat_g*p) * _np.cos(lat_s*p) * (1-_np.cos((lon_s-lon_g)*p))/2
        dist = 12742 * _np.arcsin(_np.sqrt(a))

        # get closest
        # argmin = dist.argmin()//dist.shape[1], dist.argmin()%dist.shape[1]
        # out[e,:2] = argmin        
        # out[e,2] = lon_g[argmin]
        # out[e,3] = lat_g[argmin]
        # out[e,4] = lon_s
        # out[e,5] = lat_s
        # out[e,6] = dist[argmin]
        dist_ar[e] = dist
    outd = {}
    # outd['out'] = out
    outd['dist'] = dist_ar
    return outd

def find_sites_on_grid(lon_lat_grid, sites, discard_outsid_grid = 2.2, #develop = False
#                      verbose = False, test = False,
                     ):
    """
    discard_outsid_grid: int
        maximum distance to closest gridpoint before considered outside grid and discarded. HRRR has a 3km grid -> minimum possible distance: np.sqrt(18)/2 = 2.12"""
    # get hrrr data at sites

    # sites = surfrad.network
    # discard_outsid_grid = 2.2
    ###############3
    if type(sites).__name__ == 'Network':
        sites = sites.stations._stations_list
    elif type(sites).__name__ == 'Station':
        sites = [sites]

    lon_lat_sites = _np.array([[site.lon, site.lat] for site in sites])
    out = get_closest_gridpoint(lon_lat_grid, lon_lat_sites)
    df = _pd.DataFrame(out, columns = ['argmin_x', 'argmin_y', 'lon_g', 'lat_g', 'lon_s', 'lat_s', 'dist_min'], index = [site.abb for site in sites])
    df = df[df.dist_min < discard_outsid_grid]
    return df


def find_sites_on_grid_dev(lon_lat_grid, sites, discard_outsid_grid = 2.2, develop = False
#                      verbose = False, test = False,
                     ):
    """
    discard_outsid_grid: int
        maximum distance to closest gridpoint before considered outside grid and discarded. HRRR has a 3km grid -> minimum possible distance: np.sqrt(18)/2 = 2.12"""

    if type(sites).__name__ == 'Network':
        sites = sites.stations._stations_list
    elif type(sites).__name__ == 'Station':
        sites = [sites]

    lon_lat_sites = _np.array([[site.lon, site.lat] for site in sites])
    dists = get_distance(lon_lat_grid, lon_lat_sites)['dist']
    ds = _xr.DataArray(dists, dims = ['site', 'x','y'], coords = {'site': [site.abb for site in sites]})
    
    #### drop if smallest distance is larger then ...
    dropthis = ds.site[(ds.min(dim=('x','y')) > discard_outsid_grid)]
    ds = ds.drop_sel(site = dropthis)
    return ds

def make_workplan(list_of_files = None,
                  download_files = False,
                  file_processing_state = 'raw',
                  path2folder_raw = '/mnt/telg/tmp/class_tmp/', 
                  path2interfld = '/mnt/telg/tmp/class_tmp_inter', 
                  path2resultfld = '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml'):
    """
    

    Parameters
    ----------
    list_of_files : TYPE, optional
        If this is None then the workplan for the concatination is generated 
        (the files in the intermediate folder will be used). The default is 
        None.
    file_location: str ('ftp', 'local'), optional
    file_processing_state: str ('raw', 'intermediate'), optional        
    path2folder_raw : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp/'.
    path2interfld : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp_inter'.
    path2resultfld : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml'.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    if not isinstance(path2folder_raw, type(None)):
        path2folder_raw = _pl.Path(path2folder_raw)
    path2interfld = _pl.Path(path2interfld)
    path2resultfld = _pl.Path(path2resultfld)
    if not download_files:
        if file_processing_state == 'intermediate':
            if isinstance(list_of_files, type(None)):
                df = _pd.DataFrame(list(path2interfld.glob('*.nc')), columns=['path2intermediate_file'])
            else:
                assert(False), 'programming required'
        elif file_processing_state == 'raw':
            if isinstance(list_of_files, type(None)):
                df = _pd.DataFrame(list(path2folder_raw.glob('*.nc')), columns=['path2tempfile'])
            else:
                assert(False), 'programming required'
            
            df['path2intermediate_file'] =df.apply(lambda row: path2interfld.joinpath(row.path2tempfile.name), axis = 1)
        else:
            assert(False), 'not an option'
            
    elif download_files:
        df = _pd.DataFrame(list_of_files, columns=['fname_on_ftp'])
        df['path2tempfile'] = df.apply(lambda row: path2folder_raw.joinpath(row.fname_on_ftp), axis = 1)
        df['path2intermediate_file'] =df.apply(lambda row: path2interfld.joinpath(row.fname_on_ftp), axis = 1)
    else:
        assert(False), f'{download_files} is not an option for file_location ... valid: (local, ftp)'
    # return df
    df['product_name'] = df.apply(lambda row: row.path2intermediate_file.name.split('_')[1], axis = 1)
    df['satellite'] =df.apply(lambda row: row.path2intermediate_file.name.split('_')[2], axis = 1)
    df['datetime_start'] =df.apply(lambda row: _pd.to_datetime(row.path2intermediate_file.name.split('_')[3][1:-1], format = '%Y%j%H%M%S'), axis = 1)
    df['datetime_end'] =df.apply(lambda row: _pd.to_datetime(row.path2intermediate_file.name.split('_')[4][1:-1], format = '%Y%j%H%M%S'), axis = 1)
    
    df['path2result'] = df.apply(lambda row: path2resultfld.joinpath(f'{row.product_name}_{row.satellite}_{row.datetime_start.strftime("%Y%m%d")}.nc'.replace('-', '_')), axis = 1)
    # check if final file exists
    df['result_exists'] = df.apply(lambda row: row.path2result.is_file(), axis = 1)
    df = df[~df.result_exists]
    df['intermediate_exists'] = df.apply(lambda row: row.path2intermediate_file.is_file(), axis = 1)
    # df = df[~df.intermediate_exists]
    return df

def process_workplan_row(row, ftp_settings = None, sites = None, keep_raw_file = True):
    if not isinstance(ftp_settings, type(None)):
        ftp_server = ftp_settings['ftp_server']
        ftp_login = ftp_settings['ftp_login']
        ftp_password = ftp_settings['ftp_password']
        ftp_path2files = ftp_settings['ftp_path2files']
        # local_file_source = ftp_settings['local_files_source']
    
    # download file
    if not row.path2tempfile.is_file():
        ### connect to ftp
        ftp = _ftplib.FTP(ftp_server) 
        ftp.login(ftp_login, ftp_password) 
        #         out['ftp'] = ftp
        ### navigate on ftp
        ftp.cwd(ftp_path2files)
        
        ftp.retrbinary(f'RETR {row.fname_on_ftp}', open(row.path2tempfile, 'wb').write)
        ftp.close()
    
    # open file
    try:
        classinst = _satlab.open_class_file(row.path2tempfile)
    except ValueError:
        print(f'Problems reading {row.path2tempfile} ... skip.')
        return False
    
    # find stations on grid
    sitematchtablel = find_sites_on_grid(classinst.lonlat, sites)
    
    # do the projection of variables
    ds = classinst.ds
    for e,(idx, rowsmt) in enumerate(sitematchtablel.iterrows()):
        ds_at_site = ds.isel(x= int(rowsmt.argmin_y), y=int(rowsmt.argmin_x)) #x and y seam to be interchanged
        
        # drop variables and coordinates that are not needed
        dropthis = [var.__str__() for var in ds_at_site if var.__str__() not in ['AOD', 'DQF', 'AE1', 'AE2', 'AE_DQF']] + [cor for cor in ds_at_site.coords]
        ds_at_site = ds_at_site.drop(dropthis)
        
        # add armin_x, argmin_y, etc to the dataset
        for k in rowsmt.index:
            ds_at_site[k] = rowsmt[k]
        
        # add site dimension
        ds_at_site = ds_at_site.expand_dims({'site': [rowsmt.name]})
    
        if e == 0:
            ds_at_sites = ds_at_site
        else:
            ds_at_sites = _xr.concat([ds_at_sites, ds_at_site], 'site')
    
    ### add dimentions for later concatination and save
    ds_at_sites['datetime_end'] = row.datetime_end
    ds_at_sites = ds_at_sites.expand_dims({'datetime':[row.datetime_start]})
    
    # encoding did not improve file size
    if 0:
        encoding = {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds_at_sites.variables}
        encoding['argmin_x']['dtype'] = 'int16'
        encoding['argmin_x']['_FillValue'] = -9999
        encoding['argmin_y']['dtype'] = 'int16'
        encoding['argmin_y']['_FillValue'] = -9999
        encoding['site']['dtype'] = 'object'
        encoding.pop('datetime')
        encoding.pop('datetime_end')
    else:
        encoding = None
    
    ds_at_sites.to_netcdf(row.path2intermediate_file, encoding = encoding)
    
    # delete temp file
    if not keep_raw_file:
        row.path2tempfile.unlink()
    return ds_at_sites

def process_workplan_row_dev(row, ftp_settings = None, sites = None, keep_raw_file = True):
    if not isinstance(ftp_settings, type(None)):
        ftp_server = ftp_settings['ftp_server']
        ftp_login = ftp_settings['ftp_login']
        ftp_password = ftp_settings['ftp_password']
        ftp_path2files = ftp_settings['ftp_path2files']
        # local_file_source = ftp_settings['local_files_source']
    
    # download file
    if not row.path2tempfile.is_file():
        ### connect to ftp
        ftp = _ftplib.FTP(ftp_server) 
        ftp.login(ftp_login, ftp_password) 
        #         out['ftp'] = ftp
        ### navigate on ftp
        ftp.cwd(ftp_path2files)
        
        ftp.retrbinary(f'RETR {row.fname_on_ftp}', open(row.path2tempfile, 'wb').write)
        ftp.close()
    
    # open file
    classinst = _satlab.open_class_file(row.path2tempfile)
    
    # find stations on grid
    distance_ds = find_sites_on_grid_dev(classinst.lonlat, sites)
    # sitematchtablel = 1 # just to avoid an error
    # do the projection of variables
    ds = classinst.ds
    for e,(idx, rowsmt) in enumerate(sitematchtablel.iterrows()):
        ds_at_site = ds.isel(x= int(rowsmt.argmin_y), y=int(rowsmt.argmin_x)) #x and y seam to be interchanged
        
        # drop variables and coordinates that are not needed
        dropthis = [var.__str__() for var in ds_at_site if var.__str__() not in ['AOD', 'DQF', 'AE1', 'AE2', 'AE_DQF']] + [cor for cor in ds_at_site.coords]
        ds_at_site = ds_at_site.drop(dropthis)
        
        # add armin_x, argmin_y, etc to the dataset
        for k in rowsmt.index:
            ds_at_site[k] = rowsmt[k]
        
        # add site dimension
        ds_at_site = ds_at_site.expand_dims({'site': [rowsmt.name]})
    
        if e == 0:
            ds_at_sites = ds_at_site
        else:
            ds_at_sites = _xr.concat([ds_at_sites, ds_at_site], 'site')
    
    ### add dimentions for later concatination and save
    ds_at_sites['datetime_end'] = row.datetime_end
    ds_at_sites = ds_at_sites.expand_dims({'datetime':[row.datetime_start]})
    
    # encoding did not improve file size
    if 0:
        encoding = {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds_at_sites.variables}
        encoding['argmin_x']['dtype'] = 'int16'
        encoding['argmin_x']['_FillValue'] = -9999
        encoding['argmin_y']['dtype'] = 'int16'
        encoding['argmin_y']['_FillValue'] = -9999
        encoding['site']['dtype'] = 'object'
        encoding.pop('datetime')
        encoding.pop('datetime_end')
    else:
        encoding = None
    
    ds_at_sites.to_netcdf(row.path2intermediate_file, encoding = encoding)
    
    # delete temp file
    if not keep_raw_file:
        row.path2tempfile.unlink()
    return ds_at_sites

    

def project_satellite2sites(sites,
                             download_files = False,
                             path2folder_raw = '/mnt/telg/tmp/class_tmp/',
                             path2interfld = '/mnt/telg/tmp/class_tmp_inter',
                             concatinate = 'all', 
                             path2resultfld = '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml',
                             ftp_server = 'ftp.avl.class.noaa.gov',
                             ftp_login = 'anonymous',
                             ftp_password = 'user@internet',
                             ftp_path2files = '/sub/hagen.telg/53485',
                             keep_raw_file = True,
                             no_of_cpu = 3,
                             test = False,
                             verbose = False):
    """
    This function projects satellite data (2D) to particular sites 
    on the ground. One file input file will generate 1 output file.
    So one most likely will want to concainate the files using the
    concat_projected_files function.

    Parameters
    ----------
    sites : TYPE
        DESCRIPTION.
    download_files: bool, optional
        If True all the ftp settings are required and files are
        downloaded to path2folder_raw. If False the raw data needs
        to be in path2folder_raw.    
    file_processing_state: str ('raw')
        Descirption
    path2folder_raw : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp/'.
    path2interfld : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp_inter'.
    concatinate: str
        If to concatinate file.
        'all': concatinate all
        False: no concatination
        'day': this will only concatinate days that are complete. 
            This is for cronjobs and to prevent partial days are 
            concatinated.
    path2resultfld : str, optional
        Location of the concatinated file. This is also needed even
        if concatination is not done because it will test if the
        corresponding concatinated file exist and therefore skip
        processing. The default is
        '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml'.
    ftp_server : TYPE, optional
        DESCRIPTION. The default is 'ftp.avl.class.noaa.gov'.
    ftp_login : TYPE, optional
        DESCRIPTION. The default is 'anonymous'.
    ftp_password : TYPE, optional
        DESCRIPTION. The default is 'user@internet'.
    ftp_path2files : TYPE, optional
        DESCRIPTION. The default is '/sub/hagen.telg/53485'.
    no_of_cpu : TYPE, optional
        DESCRIPTION. The default is 3.
    test : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    out = {}
    list_of_files = None
    if download_files:
    # if isinstance(path2local_files, type(None)):
        # connect to ftp
        ftp = _ftplib.FTP(ftp_server) 
        ftp.login(ftp_login, ftp_password) 
    
        ### navigate on ftp
        bla = ftp.cwd(ftp_path2files)
        if verbose:
            print(bla)
        list_of_files = ftp.nlst()
        ftp.close()
        
    workplan_full = make_workplan(list_of_files = list_of_files,
                                  download_files = download_files,
                                  file_processing_state = 'raw',
                                  path2folder_raw = path2folder_raw,
                                  path2interfld = path2interfld,
                                  path2resultfld = path2resultfld)
    
    # remove where intermediate exists
    workplan = workplan_full[~workplan_full.intermediate_exists]
    out['workplan'] = workplan.copy()
    
    ftp_settings = {}
    ftp_settings['ftp_server'] = ftp_server
    ftp_settings['ftp_login'] = ftp_login
    ftp_settings['ftp_password'] = ftp_password
    ftp_settings['ftp_path2files'] = ftp_path2files
    
    if workplan.shape[0] == 0:
        out['no_of_files_processed_bdl'] = 0
        return out

    if test:
        workplan = workplan.iloc[[test],:]
    
    if no_of_cpu == 1:    
        for idx, row in workplan.iterrows():
            ds_at_sites = process_workplan_row(row, ftp_settings, sites, keep_raw_file = keep_raw_file)
            out['ds_at_sites_last'] = ds_at_sites

    else:
        print(f'scrape_class workplan shape: {workplan.shape}')

        # ftp_settings['local_files_source'] = local_file_source
        pool = _mp.Pool(processes=no_of_cpu)
        idx, rows = zip(*list(workplan.iterrows()))
        out['pool_return'] = pool.map(_partial(process_workplan_row, **{'ftp_settings': ftp_settings, 'sites': sites, 'keep_raw_file': keep_raw_file}), rows)
        pool.close()
    
    
    
    # out_bdl = concat_scrapefiles(workplan_full)
    # out['no_of_files_processed_bdl'] = out_bdl['no_of_files_processed']
    
    out['workplan_full'] = workplan_full
    
    if concatinate == 'all':
        out['last_concat_ds'] = concat_projected_files(path2interfld=path2interfld, 
                                                path2resultfld=path2resultfld,
                                                which2concatinate='all',)
    
    return out


def concat_projected_files(path2interfld = '/mnt/telg/tmp/class_tmp_inter',
                     path2resultfld = '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml',
                     which2concatinate = 'all',
                     test = False):
    """
    Concatinates the projected files.

    Parameters
    ----------
    path2interfld : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp_inter'.
    path2resultfld : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/projects/GOES_R_ABI/data/sattelite_at_gml'.
    which2concatinate : str, optional
        If to concatinate all ('all') or just those of a complete 
        day ('day'). The latter prefents from concatination of 
        incomplet days if code is run by a cronjob. The default is 
        'all'.
    test : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    # select all rows where intermediate files exists, usually this should be all
    workplan = make_workplan(None,
                             download_files = False,
                             file_processing_state = 'intermediate',
                             path2folder_raw = None,
                             path2interfld = path2interfld,
                             path2resultfld = path2resultfld)
    
    print(f'concat_scrapefiles workplan shape: {workplan.shape}')


    # there are the files for both satellites -> group them
    goupbysat = workplan.groupby('satellite')
    no_of_files_processed = 0
    ds = None
    for sat, group_sat in goupbysat:
        group_sat = group_sat.copy()
        #remove last day ... only work on the days before last to get daily files
        group_sat.sort_values('path2result', inplace=True)
        last_day = group_sat.path2result.unique()[-1]
        
        if which2concatinate == 'day':
            group_sat = group_sat[group_sat.path2result != last_day]

        group_sat.sort_values(['datetime_start'], inplace=True)
        no_of_files_processed += group_sat.shape[0]
        for p2r,group_sat_p2r in group_sat.groupby('path2result'):
            # group_sat_p2r = group_sat_p2r.copy()
            # group_sat_p2r.sort_values('datetime')
            # print(group_sat_p2r.path2intermediate_file)
            # return group_sat_p2r
            
            # try:
                
            # for some reason this throughs an error every now an then ... the loop  does not
            # ds = _xr.open_mfdataset(group_sat_p2r.path2intermediate_file, 
            #                     # concat_dim='datetime',
            #                     )
            for e,(idx,row) in enumerate(group_sat_p2r.iterrows()):
                # print(row.path2intermediate_file)
                dst = _xr.open_dataset(row.path2intermediate_file)
                if e == 0:
                    ds = dst
                else:
                    ds = _xr.concat([ds, dst], dim = 'datetime')
                    
            # except:
            #     return group_sat_p2r
            
            assert(not p2r.is_file())
            ds.to_netcdf(p2r)
        
    return {'workplan':workplan, 'ds_last': ds, 'no_of_files_processed':no_of_files_processed}