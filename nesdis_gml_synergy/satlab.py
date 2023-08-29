import xarray as _xr
import pathlib as _pl
import numpy as _np
# import cartopy.crs as ccrs
# import metpy 
# from scipy import interpolate
# from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap as _Basemap
from pyproj import Proj as _Proj
# import urllib as _urllib
# from pyquery import PyQuery as _pq
import pandas as _pd
import matplotlib.pyplot as _plt
import mpl_toolkits.basemap as _basemap
import os as _os
# import numba as _numba
import multiprocessing as _mp
import functools as _functools
from numba import vectorize, int8, float32
import s3fs as _s3fs
import concurrent.futures
import nesdis_gml_synergy.info as ngsinf


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
                    dst = _xr.open_dataset(fn)
                    dst = dst.where(~dst.Latitude.isnull(), drop = True)
                    dst = dst.where(~dst.Longitude.isnull(), drop = True)
                    dslist.append(dst)
                ds = _xr.concat(dslist, dim = 'Rows')
            else:      
                ds = _xr.open_dataset(p2f)
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
        classinst = GeosSatteliteProducts(ds)
        return classinst

    #### VIRRS products
    if 'AEROSOL_AOD_EN' in product_name:
        pv = _np.unique([float(p.name.split('_')[1][slice(1,4,2)])/10 for p in p2f])
        assert(len(pv) == 1), f'version of files is different ({pv})'
        pv = pv[0]
        if verbose:
            print(f'Found AEROSOL_AOD_EN version {pv}')
        classinst = JRR_AOD(ds, product_version = pv)

    #### ABI products    
    elif 'ABI-L2-AODC' in product_name:
        classinst = ABI_L2_AOD(ds)
    elif product_name[:-1] == 'ABI-L2-MCMIPC-M':
        classinst = ABI_L2_MCMIPC_M6(ds)
    elif product_name[:-4] == 'ABI-L2-LST':
        classinst = ABI_L2_LST(ds)
        if verbose:
            print('identified as: ABI-L2-LSTC-M6')
    elif product_name[:-4] == 'ABI-L2-COD':
        classinst = ABI_L2_COD(ds)
        if verbose:
            print('identified as: ABI_L2_COD.')
    elif product_name[:-4] == 'ABI-L2-ACM':
        classinst = ABI_L2_ACM(ds)
        if verbose:
            print('identified as: ABI_L2_ACM.')
    elif product_name[:-4] == 'ABI-L2-ADP':
        classinst = ABI_L2_ADP(ds)
        if verbose:
            print('identified as: ABI_L2_ADP.')
    elif product_name[:-4] == 'ABI-L2-ACHA':
        classinst = ABI_L2_ACHA(ds)
        if verbose:
            print('identified as: ABI_L2_ACHA.')
    elif product_name[:-4] == 'ABI-L2-CTP':
        classinst = ABI_L2_CTP(ds)
        if verbose:
            print('identified as: ABI_L2_CTP.')
    elif product_name[:-4] == 'ABI-L2-DSR':
        classinst = ABI_L2_DSR(ds)
        if verbose:
            print('identified as: ABI_L2_DSR.')
    elif product_name == 'ABI L2 Shortwave Radiation Budget (SRB)':
        classinst = ABI_L2_SRB(ds)
        if verbose:
            print('identified as: ABI_L2_DSR.')
        
    else:
        classinst = GeosSatteliteProducts(ds)
        if verbose:
            print('not identified')
        # assert(False), f'The product {product_name} is not known yet, programming required.'
    
    if not isinstance(extent, type(None)):
        classinst  = classinst.select_area(extent)
        
    
    return classinst


class ProjectionProject(object):
    def __init__(self,sites,
                 # list_of_files = None,
                  # download_files = False,
                  # file_processing_state = 'raw',
                  path2folder_raw = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI_L2_AODC_M6_G16/', 
                  path2interfld = '/mnt/telg/tmp/class_tmp_inter', 
                  path2resultfld = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_projected/ABI_L2_AODC_M6_G16/',
                  generate_missing_folders = False):
        """
        Not clear where this is still used???

        Parameters
        ----------
        sites : TYPE
            DESCRIPTION.
        # list_of_files : TYPE, optional
            DESCRIPTION. The default is None.
        # download_files : TYPE, optional
            DESCRIPTION. The default is False.
        # file_processing_state : TYPE, optional
            DESCRIPTION. The default is 'raw'.
        path2folder_raw : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI_L2_AODC_M6_G16/'.
        path2interfld : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/tmp/class_tmp_inter'.
        path2resultfld : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/data/smoke_events/20200912_18_CO/goes_projected/ABI_L2_AODC_M6_G16/'.
        generate_missing_folders : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.sites = sites
        self.list_of_files = None
        self.download_files = False
        self.path2folder_raw = _pl.Path(path2folder_raw)
        self.path2interfld_point = _pl.Path(path2interfld).joinpath('point')
        self.path2interfld_area = _pl.Path(path2interfld).joinpath('area')
        self.path2resultfld_point = _pl.Path(path2resultfld).joinpath('point')
        self.path2resultfld_area = _pl.Path(path2resultfld).joinpath('area')
        outputfld = [self.path2interfld_point, self.path2interfld_area , self.path2resultfld_point, self.path2resultfld_area]
        if generate_missing_folders:
            for fld in outputfld:
                try:
                    fld.mkdir(exist_ok=True)
                except:
                    fld.parent.mkdir(exist_ok=True)
        
        for fld in outputfld:
            assert(fld.is_dir()), f'no such folder {fld.as_posix()}, set generate_missing_folders to true to generate folders'
        
        self._workplan = None
            
    
    @property
    def workplan(self):
        # list_of_files = None,
        #           download_files = False,
        #           file_processing_state = 'raw',
        #           path2folder_raw = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI_L2_AODC_M6_G16/', 
        #           path2interfld = '/mnt/telg/tmp/class_tmp_inter', 
        #           path2resultfld = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_projected/ABI_L2_AODC_M6_G16/'):
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
        if isinstance(self._workplan, type(None)):
            if not self.download_files:
                # if file_processing_state == 'intermediate':
                if 0:
                    pass
                    if isinstance(self.list_of_files, type(None)):
                        df = _pd.DataFrame(list(self.path2interfld.glob('*.nc')), columns=['path2intermediate_file'])
                    else:
                        assert(False), 'programming required'
                # elif file_processing_state == 'raw':
                elif 1: 
                    if isinstance(self.list_of_files, type(None)):
                        df = _pd.DataFrame(list(self.path2folder_raw.glob('*.nc')), columns=['path2tempfile'])
                    else:
                        assert(False), 'programming required'
                    
                    df['path2intermediate_file_point'] =df.apply(lambda row: self.path2interfld_point.joinpath(row.path2tempfile.name), axis = 1)
                    df['path2intermediate_file_area'] =df.apply(lambda row: self.path2interfld_area.joinpath(row.path2tempfile.name), axis = 1)
                else:
                    assert(False), 'not an option'
                    
            elif self.download_files:
                assert(False), 'this will probably not work'
                df = _pd.DataFrame(self.list_of_files, columns=['fname_on_ftp'])
                df['path2tempfile'] = df.apply(lambda row: self.path2folder_raw.joinpath(row.fname_on_ftp), axis = 1)
                df['path2intermediate_file'] =df.apply(lambda row: self.path2interfld.joinpath(row.fname_on_ftp), axis = 1)
            else:
                assert(False), f'{self.download_files} is not an option for file_location ... valid: (local, ftp)'
            # return df
            df['product_name'] = df.apply(lambda row: row.path2intermediate_file_point.name.split('_')[1], axis = 1)
            df['satellite'] =df.apply(lambda row: row.path2intermediate_file_point.name.split('_')[2], axis = 1)
            df['datetime_start'] =df.apply(lambda row: _pd.to_datetime(row.path2intermediate_file_point.name.split('_')[3][1:-1], format = '%Y%j%H%M%S'), axis = 1)
            df['datetime_end'] =df.apply(lambda row: _pd.to_datetime(row.path2intermediate_file_point.name.split('_')[4][1:-1], format = '%Y%j%H%M%S'), axis = 1)
            
        
            df['path2result_point'] = df.apply(lambda row: self.path2resultfld_point.joinpath(f'{row.product_name}_{row.satellite}_{row.datetime_start.strftime("%Y%m%d")}.nc'.replace('-', '_')), axis = 1)
            df['path2result_area'] = df.apply(lambda row: self.path2resultfld_area.joinpath(f'{row.product_name}_{row.satellite}_{row.datetime_start.strftime("%Y%m%d")}.nc'.replace('-', '_')), axis = 1)
            # check if final file exists
            df['result_exists'] = df.apply(lambda row: row.path2result_point.is_file(), axis = 1)
            df = df[~df.result_exists]
            df['intermediate_exists'] = df.apply(lambda row: row.path2intermediate_file_point.is_file(), axis = 1)
            df = df[~df.intermediate_exists]
            df.sort_values('datetime_start', inplace=True)
            self._workplan = df
        return self._workplan
    
    @workplan.setter
    def workplan(self, new_workplan):
        self._workplan = new_workplan
        
    def process(self, no_of_cpu = 3, test = False):

        # self.remove_artefacts()
        
        if test == 2:
            wpt = self.workplan.iloc[:1]
        elif test == 3:
            wpt = self.workplan.iloc[:no_of_cpu]
        else:
            wpt = self.workplan
        
        if no_of_cpu == 1:
            for idx, row in wpt.iterrows():
                wpe = WorkplanEntry(row, project = self)
                wpe.process()
        else:
            # pool = _mp.Pool(processes=no_of_cpu)
            pool = _mp.Pool(processes=no_of_cpu, maxtasksperchild=1)
            idx, rows = zip(*list(wpt.iterrows()))
            # out['pool_return'] = pool.map(partial(process_workplan_row, **{'ftp_settings': ftp_settings, 'sites': sites}), rows)
            # out = {}
            # pool_return = 
            
            # pool.map(_functools.partial(WorkplanEntry, **{'project': self, 'autorun': True}), rows)
            pool.map_async(_functools.partial(WorkplanEntry, **{'project': self, 'autorun': True}), rows)
            # out['pool_return'] = pool_return
            
            pool.close() # no more tasks
            pool.join()
        # if test == False:
        #     concat = Concatonator(path2scraped_files = self.path2data,
        #                  path2concat_files = self.path2concatfiles,)
        #     concat.save()
        #     out['concat'] = concat
        return 

class WorkplanEntry(object):
    def __init__(self, workplanrow, project = None, autorun = False):
        self.project = project
        self.row = workplanrow
        self.verbose = True
        
        # self._hrrr_inst = None
        # self._projection = None
        if autorun:
            self.process()
            
    def process(self, keep_ds = False, save = True):
        sat_inst = open_file(self.row.path2tempfile)
        proj = sat_inst.project_on_sites(self.project.sites)
        
        ds = proj.projection2area
        ds['datetime_end'] = self.row.datetime_end
        ds = ds.expand_dims({'datetime':[self.row.datetime_start]})
        if save:
            ds.to_netcdf(self.row.path2intermediate_file_area)
        if keep_ds:
            self.ds_area = ds
        
        ds = proj.projection2point
        ds['datetime_end'] = self.row.datetime_end
        ds = ds.expand_dims({'datetime':[self.row.datetime_start]})
        
        if save:
            ds.to_netcdf(self.row.path2intermediate_file_point)
        if keep_ds:
            self.ds_point = ds
        return


def process_date(date_group, date, verbose, save, test):
    try:
        ds = _xr.open_mfdataset(date_group.path2scraped_files)
        ds = ds.assign_coords(datetime=_pd.to_datetime(ds.datetime))
    except ValueError as err:
        errmsg = err.args[0]
        err.args = (f'Problem encountered while processing date {date}: {errmsg}',)
        raise

    fn_out = date_group.path2concat_files.unique()
    assert(len(fn_out) == 1), 'not possible ... I think'
    
    if verbose:
        print(f'Saving to {fn_out[0]}.')
    if save:
        ds.to_netcdf(fn_out[0])
    if test:
        return ds
    return None        

class Concatonator(object):
    def __init__(self, path2scraped_files = '/mnt/telg/tmp/class_tmp_inter/point',
                       path2concat_files = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_projected/ABI_L2_AODC_M6_G16/point',
                       datetime_format = 'ABI_L2_ACHA_projected2surfrad_%Y%m%d_%H%M%S.nc',
                       rule = 'daily',
                       skip_last_day = True,
                       file_prefix = 'concat',
                       test = False):
        
        assert(rule == 'daily'), f'Sorry, only rule="daily" works so far... programming required if you want to use "{rule}"'
        self.path2scraped_files = _pl.Path(path2scraped_files)
        self.path2concat_files = _pl.Path(path2concat_files)
        self.skip_last_day = skip_last_day
        self.datetime_format = datetime_format
        self.file_prefix = file_prefix
        try:
            self.path2concat_files.mkdir(exist_ok=True)
        except:
            self.path2concat_files.mkdir(exist_ok=True)
        
        self.test = test
        
        self._workplan = None
        self._concatenated = None
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            ## make a workplan
            workplan = _pd.DataFrame(self.path2scraped_files.glob('*.nc'), columns=['path2scraped_files'])
            
            # get datetime
            # df['datetime_start'] =df.apply(lambda row: _pd.to_datetime(row.path2intermediate_file_point.name.split('_')[3][1:-1], format = '%Y%j%H%M%S'), axis = 1)
            workplan['datetime'] =workplan.apply(lambda row: _pd.to_datetime(row.path2scraped_files.name, format = self.datetime_format), axis = 1)
           
            # if 0:
                # get the date
            #### FIXME the below results in errors (sometimes?), the below should show if the reason is ab enott workplan.
            try:
                workplan['date'] = workplan.apply(lambda row: row.datetime.date(), axis=1)
            except:
                print(f'wp shape: {workplan.shape}')
                raise

            #remove last day ... only work on the days before last to get daily files
            if self.skip_last_day:
                workplan.sort_values('date', inplace=True)
                last_day = workplan.date.unique()[-1]
                workplan = workplan[workplan.date != last_day].copy()

            # get product name
            workplan['product_name'] = workplan.apply(lambda row: row.path2scraped_files.name.split('_')[1], axis = 1)

            # output paths
            
            # workplan['path2concat_files'] = workplan.apply(lambda row: self.path2concat_files.joinpath(f'goes_at_gml_{row.product_name}_{row.date.year:04d}{row.date.month:02d}{row.date.day:02d}.nc'), axis = 1)
            workplan['path2concat_files'] = workplan.apply(lambda row: self.path2concat_files.joinpath(f'{self.file_prefix}_{row.date.year:04d}{row.date.month:02d}{row.date.day:02d}.nc'), axis = 1)

            # remove if output path exists
            workplan['p2rf_exists'] = workplan.apply(lambda row: row.path2concat_files.is_file(), axis = 1)
            workplan = workplan[ ~ workplan.p2rf_exists].copy()
            
            workplan.sort_values(['datetime'], inplace=True)
            
            workplan.index = workplan['datetime']
            
            
            self._workplan = workplan
            
        return self._workplan
    
    @workplan.setter
    def workplan(self,value):
        self._workplan = value
    
    def concat_and_save(self, save = True, test = False, verbose = False, num_cpus=None):
        """
        Processes the workplan.

        Parameters
        ----------
        save : bool, optional
            For testing, False will skip the saving; still returns the DataSet. The default is True.
        test : bool, optional
            If True only the first line will be processed. The default is False.
        verbose : bool, optional
            Some info along the way. The default is False.

        Returns
        -------
        The last concatinated xarray.DataSet.

        """
        # def process_date(date_group):
        #     try:
        #         ds = _xr.open_mfdataset(date_group.path2scraped_files)
        #         ds = ds.assign_coords(datetime=_pd.to_datetime(ds.datetime))
        #     except ValueError as err:
        #         errmsg = err.args[0]
        #         err.args = (f'Problem encountered while processing date {date}: {errmsg}',)
        #         raise
        
        #     fn_out = date_group.path2concat_files.unique()
        #     assert(len(fn_out) == 1), 'not possible ... I think'
            
        #     if verbose:
        #         print(f'Saving to {fn_out[0]}.')
        #     if save:
        #         ds.to_netcdf(fn_out[0])
        #     if test:
        #         return ds
        #     return None
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = []
            for date, date_group in self.workplan.groupby('date'):
                if date == self.workplan.iloc[-1].date:
                    break
                futures.append(executor.submit(process_date, date_group, date, verbose, save, test))
                
                if test:
                    break
            
            # Retrieve results from futures (if test=True)
            results = [future.result() for future in futures if future.result() is not None]
            
        if verbose:
            print('Done')
        
        # Return the last concatenated xarray.DataSet
        return results[-1] if results else None
        
        
        
        # for date,date_group in self.workplan.groupby('date'):
        #     #### interrupt if last date to avoid processing incomplete days
        #     if date == self.workplan.iloc[-1].date:
        #         break
        #     try:
        #         # ds = xr.concat(fc_list, dim = 'datetime')
        #         ds = _xr.open_mfdataset(date_group.path2scraped_files)#, concat_dim='datetime')
        #         ds = ds.assign_coords(datetime = _pd.to_datetime(ds.datetime))  #there was an issue with the format of the time koordinate. This statement can probably be removed at some point (20230420)
        #     except ValueError as err:
        #         errmsg = err.args[0]
        #         err.args = (f'Problem encontered while processing date {date}: {errmsg}',)
        #         raise

        #     fn_out = date_group.path2concat_files.unique()
        #     assert(len(fn_out) == 1), 'not possible ... I think'
        #     # concat.append({'dataset': ds, 'fname': fn_out[0]})
        #     if verbose:
        #         print(f'Saving to {fn_out[0]}.')
        #     if save:
        #         ds.to_netcdf(fn_out[0])  
        #     if test:
        #         break
        # if verbose:
        #     print('Done')
        # return ds
    
    
    def deprecated_concat_and_save(self, save = True, test = False, verbose = False):
        """
        Processes the workplan.

        Parameters
        ----------
        save : bool, optional
            For testing, False will skip the saving; still returns the DataSet. The default is True.
        test : bool, optional
            If True only the first line will be processed. The default is False.
        verbose : bool, optional
            Some info along the way. The default is False.

        Returns
        -------
        The last concatinated xarray.DataSet.

        """
        for date,date_group in self.workplan.groupby('date'):
            #### interrupt if last date to avoid processing incomplete days
            if date == self.workplan.iloc[-1].date:
                break
            try:
                # ds = xr.concat(fc_list, dim = 'datetime')
                ds = _xr.open_mfdataset(date_group.path2scraped_files)#, concat_dim='datetime')
                ds = ds.assign_coords(datetime = _pd.to_datetime(ds.datetime))  #there was an issue with the format of the time koordinate. This statement can probably be removed at some point (20230420)
            except ValueError as err:
                errmsg = err.args[0]
                err.args = (f'Problem encontered while processing date {date}: {errmsg}',)
                raise

            fn_out = date_group.path2concat_files.unique()
            assert(len(fn_out) == 1), 'not possible ... I think'
            # concat.append({'dataset': ds, 'fname': fn_out[0]})
            if verbose:
                print(f'Saving to {fn_out[0]}.')
            if save:
                ds.to_netcdf(fn_out[0])  
            if test:
                break
        if verbose:
            print('Done')
        return ds
    
           

class SatelliteMovie(object):
    def __init__(self, 
                 path2fld_sat = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI-L2-MCMIP/',
                 path2fld_fig = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI-L2-MCMIP_figures/',
                 path2fld_movies = '/mnt/telg/data/smoke_events/20200912_18_CO/goes_raw/ABI-L2-MCMIP_movies/',
                 ):
        

        self.path2fld_sat = _pl.Path(path2fld_sat)
        self.path2fld_fig = _pl.Path(path2fld_fig)
        self.path2fld_movies = _pl.Path(path2fld_movies)
        
#         path2fld_fig.mkdir(exist_ok=True)
        # properties
        self._workplan = None
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            workplan = _pd.DataFrame(self.path2fld_sat.glob('*.nc'), columns=['path2sat'])
            workplan.index = workplan.apply(lambda row: _pd.to_datetime(row.path2sat.name.split('_')[-3][1:-1], format = '%Y%j%H%M%S'), axis = 1)
            workplan['path2fig'] = workplan.apply(lambda row: self.path2fld_fig.joinpath(row.path2sat.name.replace('.nc','.png')), axis = 1)
            workplan.sort_index(inplace=True)
            self._workplan = workplan
        return self._workplan
    
    @workplan.setter
    def workplan(self, new_workplan):
        self._workplan = new_workplan
        return
        
    def plot_single(self, 
                    resolution          = 'i',
                    width               = 4.5e6,
                    height              = 2.7e6,
                    lat_0               = 40.12498,
                    lon_0               = -99.2368,
                    costlines           = True,
                    mapscale            = True,
                    row                 = 0,
                    sites               = None,
                    gamma               = 2.3,
                    contrast            = 200,
                    first               = True,
                    save                = True,
                    dpi                 = 300,
                    overwrite           = True,
                    use_active_settings = False,):
        
        # read the file and make the time stamp
        if not type(row) == _pd.Series:
            row = self.workplan.iloc[row]
            
        if use_active_settings:
            resolution = self._a_resolution
            width      = self._a_width     
            height     = self._a_height    
            lat_0      = self._a_lat_0     
            lon_0      = self._a_lon_0        
            sites      = self._a_sites     
            dpi        = self._a_dpi     
            costlines  = self._a_costlines
            mapscale = self._a_mapscale
            gamma = self._a_gamma
            contrast = self._a_contrast

        else:
            self._a_resolution = resolution
            self._a_width      = width     
            self._a_height     = height    
            self._a_lat_0      = lat_0     
            self._a_lon_0      = lon_0        
            self._a_sites      = sites     
            self._a_dpi        = dpi 
            self._a_costlines  = costlines
            self._a_mapscale  = mapscale
            self._a_gamma = gamma
            self._a_contrast = contrast
            
            
        mcmip = open_file(row.path2sat)
        timestamp = _pd.to_datetime(mcmip.ds.attrs['time_coverage_start'])
        tstxt = f'{timestamp.date()}\n{timestamp.time()}'.split('.')[0]
        
        if first:
            # make the basemap instanc 
            bmap = _basemap.Basemap(resolution=resolution, projection='aea', 
                                   area_thresh=5000, 
                                         width=width, height=height, 
                #                          lat_1=38.5, lat_2=38.5, 
                                         lat_0=lat_0, lon_0=lon_0)
            bmap.drawstates(zorder = 1)
            if costlines:
                bmap.drawcoastlines(linewidth = 0.5, zorder = 1)
            bmap.drawcountries(linewidth = 1, zorder = 1)
            a = _plt.gca()
            if sites:
                a, _ = sites.plot(bmap = bmap, projection='aea',
            #                             width = width, height = height,
                                        background=None, 
                            station_symbol_kwargs={'markerfacecolor':'none', 
                                                 'markersize':5, 
                                                 'markeredgewidth':0.5},
                            station_label_format='')
            self._bmap_active = bmap
        else:
            bmap = self._bmap_active
            a = _plt.gca()
            self._txt_active.remove()
            self._pc_active.remove()
            
        mcmip.generate_rgb(gamma = gamma,
                            contrast = contrast,
                           )
        out = mcmip.plot_true_color(bmap = bmap, 
                              # contrast = 200, gamma=2.3,
                              zorder = 0,
                             )
        if out == False:
            self._pc_active = txt = a.text(0,0,'', transform=a.transAxes, zorder = 10,
                                             color = 'black',#colors[1], 
#                                              bbox=dict(facecolor=[1,1,1,0.7], linewidth = 0)
                                          )
            
        else:
            self._pc_active = out['pc']
            
        txt = a.text(0.05,0.85,tstxt, transform=a.transAxes, zorder = 10,
                     color = 'black',#colors[1], 
                     bbox=dict(facecolor=[1,1,1,0.7], linewidth = 0))
        self._txt_active = txt
        
        f = _plt.gcf()
        f.patch.set_alpha(0)
        
        #### mapscale
        if mapscale:
            sfrac = 3
            rel_x = 0.7  # 10% from the left
            rel_y = 0.1  # 10% from the bottom
            
            lon = bmap.llcrnrlon + rel_x * (bmap.urcrnrlon - bmap.llcrnrlon)
            lat = bmap.llcrnrlat + rel_y * (bmap.urcrnrlat - bmap.llcrnrlat)
            
            sw = int(width/sfrac/100000) * 100
            bmap.drawmapscale(lon, lat, lon, lat, sw, barstyle='fancy', units='km', 
                              # fontsize=9, 
                              # yoffset=20000,
                             )
        
        if save:
            if row.path2fig.is_file():
                if not overwrite:
                    print(f'file exists, saving skipped! ({row.path2fig})')
                    return
            f.savefig(row.path2fig, dpi = dpi, bbox_inches = 'tight')
            
        return bmap
        
    def create_images(self, overwrite = False, use_active_settings = True, verbose = False):
        first = True
        if not overwrite:
            workplan = self.workplan[~self.workplan.path2fig.apply(lambda x: x.is_file())]
        else:
            workplan = self.workplan
        print(f'Number of files to process: {workplan.shape[0]}')
        for e,(idx, row) in enumerate(workplan.iterrows()):
            if verbose:
                print(row.path2sat)
            self.plot_single(row = row, first = first, overwrite = overwrite, use_active_settings = use_active_settings)
            first = False
#             if e == 2:
#                 break
    def create_movies(self, overwrite = False, name = ''):
        dates = self.workplan.apply(lambda row: row.name.date(), axis = 1)
        groups = self.workplan.groupby(dates)
        for date,grp in groups:
            path2file_movie = self.path2fld_movies.joinpath(f'{name}_{date}.mp4'.replace('-','_'))
            path2file_movie_file_list = self.path2fld_movies.joinpath(f'{name}_{date}.txt'.replace('-','_'))
            if path2file_movie.is_file():
                if overwrite:
                    path2file_movie.unlink()
                else:
                    print(f'Movie file exists, skip! ({path2file_movie})')

            # generate the file that contains the list of files that go into the movie
            with open(path2file_movie_file_list, mode = 'w') as raus:
                for idx, row in grp.iterrows():
            #         pass
                    raus.write(f"file '{row.path2fig}'\n")

            fps = 5
            command = "ffmpeg -r {fps} -f concat -safe 0 -i '{mylist}'  '{pathout}'".format(fps=fps,
                                                                                mylist = path2file_movie_file_list,
            #                                                                     folder=save_figures2folder,
                                                                                pathout=path2file_movie)
            out = _os.system(command)
            if out != 0:
                print('something went wrong with creating movie from saved files (error code {}).\n command:\n{}'.format(out, command))

# class SatelliteDataQuery(object):
#     """ This is an older way to download data from the AWS. It from a time when I did not undstand that this webpage also just directs you to the AWS"""
#     def __init__(self):
#         self._base_url = 'http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/goes16_download.cgi'
#         self.path2savefld = _pl.Path('/mnt/data/data/goes/')
#         self.path2savefld.mkdir(exist_ok=True)
#         html = _urllib.request.urlopen(self._base_url).read()
#         self._doc = _pq(html)
#         self.query = _pd.DataFrame(columns=['source', 'satellite', 'domain', 'product', 'date', 'hour', 'minute', 'url2file', 'url_inter','inbetween_page'])
#         self._url_inter = _pd.DataFrame(columns=['url', 'sublinks'])
#     @property
#     def available_products(self):
#         # available products
#         products = [i for i in self._doc.find('select') if i.name == 'product'][0]
#         products_values = [i.attrib["value"]for  i in products]
# #         print('\n'.join([f'{i.attrib["value"]}:\t {i.text}' for  i in products]))
#         return products_values
#     @property
#     def available_domains(self):
#         domains = [i for i in self._doc.find('select') if i.name == 'domain'][0]
#         domains_values = [i.attrib["value"]for  i in domains]
# #         print('\n'.join([f'{i.attrib["value"]}:\t {i.text}' for  i in domains]))
#         return domains_values

#     @property
#     def available_satellites(self):
#         # available satellites
#         satellites = [i for i in self._doc.find('input') if i.name == 'satellite']
#         satellites_values = [i.attrib['value'] for i in satellites] #[i.attrib["value"][-2:] for i in satellites]
# #         print('\n'.join([f'{i.attrib["value"][-2:]}: {i.attrib["value"]}' for  i in satellites]))
#         return satellites_values

#     def _attache_intermeidate_linkes(self):
#         get_inter_url = lambda row: self._base_url + '?' + '&'.join([f'{i[0]}={i[1]}' for i in row.reindex(['source', 'satellite', 'domain', 'product', 'date', 'hour']).items()])
#         self.query['url_inter'] = self.query.apply(get_inter_url, axis = 1)

#     def _get_intermediate_pages(self):
#         for idx,row in self.query.iterrows():
#             intermediate_url =  row.url_inter
#         #     break

#             if intermediate_url not in self._url_inter.url.values:

#                 html_inter = _urllib.request.urlopen(intermediate_url).read()
#                 doc_inter = _pq(html_inter)

#                 sub_urls = []
#                 for link in doc_inter('a'):
#                 #     print(link.attrib['href'])
#                     if 0:
#                         if 'noaa-goes16.s3' in link.attrib['href']:
#                             print(link.attrib['href'])
#                             break
#                     else:
#                         if len(link.getchildren()) == 0:
#                             continue
#                         if not 'name' in link.getchildren()[0].attrib.keys():
#                             continue

#                         if link.getchildren()[0].attrib['name'] == 'fxx':
#                             sub_urls.append(link.attrib['href'])
#                 #             print(link.attrib['href'])
#                 #             break

#                 sub_urls = _pd.DataFrame(sub_urls, columns = ['urls'])

#                 sub_urls['datetime']= sub_urls.apply(lambda row: _pd.to_datetime(row.urls.split('/')[-1].split('_')[-3][1:-3], format = '%Y%j%H%M'), axis=1)


#                 self._url_inter = self._url_inter.append(dict(url = intermediate_url,
#                                            sublinks = sub_urls), ignore_index= True)
#                 assert(not isinstance(row.minute, type(None))), 'following minutes are available ... choose! '+ ', '.join([str(i.minute) for i in sub_urls.datetime])

#             else:
#                 pass
# #                 print('gibs schon')
                
#     def _get_link2files(self):
#         for idx,row in self.query.iterrows():
#             sublinks = self._url_inter[self._url_inter.url == row.url_inter].sublinks.iloc[0]
#             for sidx, srow in sublinks.iterrows():
#                 if srow.datetime.minute == int(row.minute):
#                     row.url2file = srow.urls
                       
#     def _generate_save_path(self):
#         def gen_output_path(self,row):
#             fld = self.path2savefld.joinpath(row['product']) 
#             fld.mkdir(exist_ok=True)
            
#             try: 
#                 p2f = fld.joinpath(row.url2file.split('/')[-1])
#             except:
#                 print(f'promblem executing " p2f = fld.joinpath(row.url2file.split('/')[-1])" in {row}, with {fld}')
#                 assert(False)
            
#             return p2f
#         self.query['path2save'] = self.query.apply(lambda row: gen_output_path(self,row), axis = 1)
    
#     @property
#     def workplan(self):
#         self._get_link2files()
#         self._generate_save_path()
#         self.query['file_exists'] = self.query.apply(lambda row: row.path2save.is_file(), axis = 1)
#         self._workplan = self.query[~self.query.file_exists]
#         return self._workplan
        
#     def add_query(self, source = 'aws',
#                      satellite = 'noaa-goes16',
#                      domain = 'C',
#                      product = 'ABI-L2-AOD',
#                      date = '2020-06-27',
#                      hour = 20,
#                      minute = [21, 26]):
        
#         if not isinstance(minute, list):
#             minute = [minute]
#         if not isinstance(hour, list):
#             hour = [hour]
            
#         for qhour in hour:
#             for qmin in minute:
#                 qdict = dict(source = source,
#                          satellite = satellite,
#                          domain = domain,
#                          product = product,
#                          date = date,
#                          hour = f'{qhour:02d}',
#                          minute = f'{qmin:02d}'
#                          )
#                 self.query = self.query.append(qdict, ignore_index = True)
#         assert(satellite in self.available_satellites)
#         assert(domain in self.available_domains)
#         assert(product in self.available_products)
        
#         self._attache_intermeidate_linkes()
#         self._get_intermediate_pages()
#         # drop line if minute is None
#         for idx, row in self.query.iterrows():
#             if isinstance(row.minute, type(None)):
#                 self.query.drop(idx, inplace=True)
                
#     def download_query(self, test = False):
#         for idx, row in self.workplan.iterrows():
#             print(f'downloading {row.url2file}', end = ' ... ')
#             if row.path2save.is_file():
#                 print('file already exists ... skip')
#             else:
#                 _urllib.request.urlretrieve(row.url2file, filename=row.path2save)
#                 print('done')
#             if test:
#                 break


class GeosSatteliteProducts(object):
    def __init__(self,file, product_version = None, verbose = False):
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        product_version : TYPE, optional
            In some cases (an error will tell you when) the product version is not readily available!. The default is None.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if type(file) == _xr.core.dataset.Dataset:
            ds = file
        else:
            assert(False), 'DEPRECATED! Use open_file!'
            ds = _xr.open_dataset(file)
        # different products give different identifies. Also the case varies, thats why some code looks convoluted
        if 'dataset_name' in ds.attrs.keys(): 
            long_name = ds.attrs['dataset_name']
            product_name = long_name.split('_')[1]
        elif 'title'in [k.lower() for k in ds.attrs.keys()]:    
            product_name = long_name = ds.attrs[list(ds.attrs.keys())[[k.lower() for k in ds.attrs.keys()].index('title')]]
        else:
            assert(False), f'neither dataset_name nor title in attr keys. Options are: {ds.attrs.keys()}'
        self.long_name = long_name
        if long_name in ngsinf.VIIRS_product_info.keys():
        # orpit_type =  
            prin = {}
            prin['sensor'] = 'VIIRS'
            if type(file) == _xr.core.dataset.Dataset:
                if isinstance(product_version, type(None)):
                    assert(False), 'Product version is given only in file name, which was not known to when dataset is passed. Pass value to product_version kwarg. In some cases this error can be bypassed by giving a -1.'
                else:
                    prin['version'] = product_version
            self.product_info = prin
        else:
            tl = ['sensor', 'level', 'name', 'version']
            self.product_info = {tl[e]:p for e,p in enumerate(product_name.split('-'))}
        
        
        self.qf_managment = None
#         self._varname4test = 'CMI_C02'

        # self.valid_qf = None
        self._lonlat = None
        
        # accomodate inconsistancies in the coordinate names
        if 'Latitude' in ds.coords:
            ds = ds.rename({'Latitude': 'lat'})
        if 'Longitude' in ds.coords:
            ds = ds.rename({'Longitude': 'lon'})
        if 'Lon' in ds.coords:
            ds = ds.rename({'Lon': 'lon'})
        if 'Lat' in ds.coords:
            ds = ds.rename({'Lat': 'lat'})
        
        if 'QCAll' in ds.variables:
            # qcvar = [v for v in self.ds.variables if v in ['DQF', 'QCAll']]
            # assert(len(qcvar) == 1), f'something wrong with the qc flag name! found {qcvar}'
            ds = ds.rename({'QCAll': 'DQF'})
        
        # same for dims
        if 'Rows' in ds.dims:
            ds = ds.rename({'Rows':'y', 'Columns':'x'})
        
        # some file are provide in scan_angle others in lat lon
        if 'x' in ds.coords:
            self.grid_type = 'scan_angle'
        # elif 'lat' in ds.coords:
        elif 'lat' in ds.coords:
            if len(ds.lat.shape) == 1: 
                self.grid_type = 'lonlat'
            elif len(ds.lat.shape) == 2:
                self.grid_type = 'lonlatmesh' # here the conversion from scan_angle to lat lon has been done by the 
        else:
            assert(False), f'It looks like there are no adequate coordinates (x,y or lon, lat) available. What is provided: {list(ds.coords)}'
        
        self.ds = ds

    @property
    def valid_2D_variables(self):      
        if self.grid_type in ['scan_angle', 'lonlatmesh']:
            var_sel = [var for var in self.ds.variables if self.ds[var].dims == ('y', 'x')]
            var_sel = [var for var in var_sel if not 'DQF' in var]
            # lat and lon might be present if lonlat was executed prior to this ... remove it
            if 'lon' in var_sel:
                var_sel.pop(var_sel.index('lon'))
                var_sel.pop(var_sel.index('lat'))
            
        elif self.grid_type == 'lonlat':
            var_sel = [var for var in self.ds.variables if self.ds[var].dims == ('lat', 'lon')]
            var_sel = [var for var in var_sel if not 'DQF' in var]
        else:
            assert(False), 'moep'
        return var_sel #list(ds.variables)

    @property
    def data_by_quality_high(self):
        return self.get_data_by_quality(['high',])
    
    @property
    def data_by_quality_high_medium(self):
        return self.get_data_by_quality(['high','medium'])
    
    @property
    def data_by_quality_high_medium_low(self):
        return self.get_data_by_quality(['high','medium', 'low'])
    
    @property
    def data_by_quality_medium(self):
        return self.get_data_by_quality(['medium',])
    
    @property
    def data_by_quality_low(self):
        return self.get_data_by_quality(['low',])
    
    def get_data_by_quality(self, quality):
        assert(isinstance(quality, list)), 'quality has to be a list of qualities'
        # select relevant variablese ... those with x and y
        # var_sel = [var for var in self.ds.variables if self.ds[var].dims == ('y', 'x')]
        
        # # we don't want DQF i think
        # var_sel.pop(var_sel.index('DQF'))
        # # lat and lon might be present if lonlat was executed prior to this ... remove it
        # if 'lon' in var_sel:
        #     var_sel.pop(var_sel.index('lon'))
        #     var_sel.pop(var_sel.index('lat'))
        var_sel = self.valid_2D_variables
        
        ds = _xr.Dataset()
        for var in var_sel:
            if self.qf_managment.qf_by_variable[var] == 'ignore':
                continue
            valid_qf = []
            for qual in quality:
                valid_qf += self.qf_managment.qf_by_variable[var][qual]
            
            # global qc variable names vary, I hope this helps
            # qcvar = [v for v in self.ds.variables if v in ['DQF', 'QCAll']]
            # assert(len(qcvar) == 1), f'something wrong with the qc flag name! found {qcvar}'
            # qcvar = qcvar[0]
            
            ds[var] = self.ds[var].where(self.ds.DQF.isin(valid_qf))
        
        #### cleanup the  coordinates
        coords2del = list(ds.coords)
        if self.grid_type in ['scan_angle',]:
            coords2del.remove('x')
            coords2del.remove('y')
        elif self.grid_type in ['lonlat', 'lonlatmesh']:
            coords2del.remove('lat')
            coords2del.remove('lon')
        else:
            assert(False), f'New product attempt?!? grid_type:{self.grid_type}'
        
        if 't' in self.ds.coords:
            coords2del.remove('t')
        ds = ds.drop_vars(coords2del)
        return ds
        
    def select_area(self, extent):
        lon_min, lon_max, lat_min, lat_max = extent
        lon, lat = self.lonlat

        # where_lon = _np.logical_and(lon > lon_min, lon < lon_max)
        # where_lat = _np.logical_and(lat > lat_min, lat < lat_max)
        where = _np.logical_and(_np.logical_and(lon > lon_min, lon < lon_max), _np.logical_and(lat > lat_min, lat < lat_max))
        
        # self.ds['lon'] = _xr.DataArray(lon, dims = ['y', 'x'])
        # self.ds['lat'] = _xr.DataArray(lat, dims = ['y', 'x'])
        
        self.ds['where_cond'] = _xr.DataArray(where, dims = ['y', 'x'])
        
        ds_sel = self.ds.where(self.ds.where_cond)
        # the above 
        ds_sel = ds_sel.dropna('x', how = 'all')
        ds_sel = ds_sel.dropna('y', how = 'all')
        ds_sel = ds_sel.drop(['lat', 'lon'])
        return open_file(ds_sel)
    
    @property
    def lonlat(self):
        if isinstance(self._lonlat, type(None)):
            # There are a few products with lat lon in it!!! First without:
            if self.grid_type == 'scan_angle':
                assert('lat' not in self.ds.variables), 'This should not exist, this would mean there are both x,y and lat, lon'
                # Satellite height
                sat_h = self.ds['goes_imager_projection'].perspective_point_height
    
                # Satellite longitude
                sat_lon = self.ds['goes_imager_projection'].longitude_of_projection_origin
    
                # Satellite sweep
                sat_sweep = self.ds['goes_imager_projection'].sweep_angle_axis
    
                # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
                # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
                x = self.ds['x'][:] * sat_h
                y = self.ds['y'][:] * sat_h
    
                # Create a pyproj geostationary map object to be able to convert to what ever projecton is required
                p = _Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    
                # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
                # to latitude and longitude values.
                XX, YY = _np.meshgrid(x, y)
                lons, lats = p(XX, YY, inverse=True)
                
                # Assign the pixels showing space as a single point in the Gulf of Alaska
    #             where = _np.isnan(self.ds[self._varname4test].values)
                where = _np.isinf(lons)
                lats[where] = 57
                lons[where] = -152
    
                self._lonlat = (lons, lats) #dict(lons = lons, 
    #                                     lats = lats)
                self.ds['lon'] = _xr.DataArray(lons, dims = ['y', 'x']).astype(_np.float32)
                self.ds['lat'] = _xr.DataArray(lats, dims = ['y', 'x']).astype(_np.float32)
            
            elif self.grid_type == 'lonlatmesh':
                self._lonlat = (self.ds.lon.values, self.ds.lat.values)
            # The lat lon ... at least in the case I had (DSR), only gave one dimentional lat lon -> meshgrid    
            elif self.grid_type == 'lonlat':
                self._lonlat = _np.meshgrid(self.ds.lon.values, self.ds.lat.values)
                # assert(False), 'noet'
            else:
                assert(False), f'noet, gridtype {self.grid_type} unkown'
        return self._lonlat
    
    def project_on_sites(self, sites):
        site_projection = Grid2SiteProjection(self, sites) 
        return site_projection
    

    
    def get_resolution(self, site = [-105.2368, 40.12498]):
        """
        Get the satellite resolotion at a particular coordinate.

        Parameters
        ----------
        at_site : iterable, or atmPy.general.measurement_site.Station instance
            This needs to be coordinates (lon, lat) or a measurment site 
            instance that has lat and lon attributes. Default are coordinates
            of Table Mountain, CO ([-105.2368, 40.12498])

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        # site = at_site
        if hasattr(site, 'lat') and hasattr(site, 'lon'):
            pass
        else:
            site = _np.array([site])
        out = self.get_closest_gridpoint(site)
    
        closest_point = out['closest_point'].iloc[0]
        dist = out['last_distance_grid']
    
        closest_point
    
        ax, ay = int(closest_point.argmin_x), int(closest_point.argmin_y)
    
        # dist[ax+1: ax+2, ay]
    
        bla = 10
        resx = ((dist[ax-bla: ax-bla+1, ay]/bla) + (dist[ax+bla: ax+bla+1, ay]/bla))/2
        resy = ((dist[ax, ay-bla: ay-bla+1]/bla) + (dist[ax, ay+bla: ay+bla+1]/bla))/2
        out = _pd.DataFrame({'lon':resx, 'lat': resy})
        return out
    
    def plot(self, variable, 
             # valid_qf = True, 
             # qf = None,
             data_quality = None,
             bmap = None, colorbar = True, **pcolor_kwargs):
        """
        Plot on map

        Parameters
        ----------
        variable : str
            which variable to plot.
        data_quality: int or list of int
            0-high, 1-medium, 2-low, some combinations are possible in form of a list.
        bmap : TYPE, optional
            DESCRIPTION. The default is None.
        **pcolor_kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        bmap : TYPE
            DESCRIPTION.
        pc : TYPE
            DESCRIPTION.
        cb : TYPE
            DESCRIPTION.

        """
        lons,lats = self.lonlat
        
        if isinstance(bmap, type(None)):
            bmap = _Basemap(resolution='c', projection='aea', area_thresh=5000, 
                     width=3000*3000, height=2500*3000, 
        #                          lat_1=38.5, lat_2=38.5, 
                     lat_0=38.5, lon_0=-97.5)
    
            bmap.drawcoastlines()
            bmap.drawcountries()
            bmap.drawstates()
            
        ### select valid quality flags given by particular product ... if defined
        if isinstance(data_quality, list):
            data_quality.sort()
            if len(data_quality) == 1:
                data_quality = data_quality[0]
            
        
        if isinstance(data_quality, type(None)):
            ds = self.ds
        elif data_quality == 0:
            ds = self.data_by_quality_high
        elif data_quality == [0,1]:
            ds = self.data_by_quality_high_medium
        elif data_quality == [0,1,2]:
            ds = self.data_by_quality_high_medium_low
        elif data_quality == 1:
            ds = self.data_by_quality_medium
        elif data_quality == [1,2]:
            ds = self.data_by_quality_medium_low
        elif data_quality == 2:
            ds = self.data_by_quality_low
        else:
            assert(False), f'{data_quality} not an option for data_quality. Choose one or a combination of [0,1,2]'
        # if valid_qf:
            
            # if not isinstance(self.valid_qf, type(None)): 
            #     ds = ds.where(ds.DQF.isin(self.valid_qf))
        # if not isinstance(qf, type(None)) :
        #     ds = ds.where(ds.DQF == qf)
        pc = bmap.pcolormesh(lons, lats, ds[variable], latlon=True, **pcolor_kwargs)
        if colorbar:
            f = _plt.gcf()
            cb = f.colorbar(pc)
            cb.set_label(ds[variable].long_name)
        else:
            cb = None
        return bmap,pc,cb

# I don't see much improvement with numba!
# @_numba.jit(nopython=True, 
#             fastmath=True, 
#             # parallel=True,
#             )
def get_dists(lon_lat_grid, lon_lat_sites):
    out = _np.zeros((lon_lat_sites.shape[0], 7), dtype = _np.float32)
    dist_array = _np.zeros(lon_lat_grid[0].shape + (len(lon_lat_sites),), dtype = _np.float32)
    lon_g, lat_g = lon_lat_grid
    for e,site in enumerate(lon_lat_sites):
        lon_s, lat_s = site

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
        dist_array[:,:,e] = dist
    return out, dist_array

class Grid2SiteProjection(object):
    def __init__(self, grid, sites):
        self.grid = grid
        self.sites = sites
        self.radii = [5,10,25,50,100]
        
        self._closest_points = None
        self._distance_grids =  None
        self._projection2poin = None
        self._projection2area = None
    
    @property
    def projection2point(self):
        if isinstance(self._projection2poin, type(None)):
            if self.grid.grid_type in  ['scan_angle', 'lonlatmesh']:
                coord1, coord2 = 'x', 'y'
            elif self.grid.grid_type in  ['lonlat', ]:
                coord1, coord2 = 'lon', 'lat'
            # select relevant variablese ... those with x and y
            # var_sel = [var for var in self.grid.ds.variables if self.grid.ds[var].dims == ('y', 'x')]
            var_sel = self.grid.valid_2D_variables + ['DQF',]
            
            ds = self.grid.ds[var_sel]
            
            # cleanup the the coordinates
            
            coords2del = list(ds.coords)
            
            try:
                coords2del.remove(coord1)
                coords2del.remove(coord2)
            except:
                pass
            # coords2del.remove('t')
            
            ds = ds.drop_vars(coords2del)
            
            for e,(idx, rowsmt) in enumerate(self.closest_grid_points.iterrows()):
                # ds_at_site = ds.isel(x= int(rowsmt.argmin_y), y=int(rowsmt.argmin_x)) #x and y seam to be interchanged
                isel_dict = {coord1: int(rowsmt.argmin_y), coord2: int(rowsmt.argmin_x)}
                ds_at_site = ds[isel_dict] #x and y seam to be interchanged
                # drop variables and coordinates that are not needed
            #     dropthis = [var.__str__() for var in ds_at_site if var.__str__() not in ['AOD', 'DQF', 'AE1', 'AE2', 'AE_DQF']] + [cor for cor in ds_at_site.coords]
            #     ds_at_site = ds_at_site.drop(dropthis)
            
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
            # ds_at_sites['datetime_end'] = row.datetime_end
            # ds_at_sites = ds_at_sites.expand_dims({'datetime':[row.datetime_start]})
            
            # clean up coordinates
            try:
                ds_at_sites = ds_at_sites.drop_vars([coord1, coord2])
            except:
                pass
            
            #### assess DQF
            qf_by_variable = self.grid.qf_managment.qf_by_variable
            variables = list(ds_at_sites.variables)

            for var in qf_by_variable:
                if qf_by_variable[var] == 'ignore':
                    # variables.pop(variables.index(var))
                    continue
                # add DQF assessed variable and set to nans
                varname = f'{var}_DQF_assessed'
                dsdqfass= ds_at_sites.DQF.copy()
                dsdqfass[:] = _np.nan
            
                # this is just for the reorganization of the variables
                variables.insert(variables.index(var)+1, varname)
            
                # set the assest DQF values
                dsdqfass[ds_at_sites.DQF.isin(qf_by_variable[var]['high'])] = 0
                if 'medium' in qf_by_variable[var].keys():
                    dsdqfass[ds_at_sites.DQF.isin(qf_by_variable[var]['medium'])] = 1
                
                if 'low' in qf_by_variable[var].keys():
                    dsdqfass[ds_at_sites.DQF.isin(qf_by_variable[var]['low'])] = 2
                    
                dsdqfass[ds_at_sites.DQF.isin(qf_by_variable[var]['bad'])] = 3
                
                # add some attributes
                dsdqfass.attrs = {}
                dsdqfass.attrs['long_name'] = 'Assessed quality flag. This created by the nesdis_gml_synergy package so simplify quality flags.'
                dsdqfass.attrs['values'] = [0,1,2,3]
                dsdqfass.attrs['meaning'] = '0-high_quality 1-medium_quality 2-low_quality 3-bad'
                
                # add to dataset
                ds_at_sites[varname] = dsdqfass.astype(_np.int8)
            
            
            # reorganize variables for user convenience
            ds_at_sites = ds_at_sites[variables]
            
            
            self._projection2poin = ds_at_sites
        return self._projection2poin
    
    def _project2area(self, data_quality, the_new_way = True):
        """
        project data to an area around the grid point

        Parameters
        ----------
        data_quality : TYPE
            Which satellite data quality to consider. 
            Options: good, intermediate
            Note: This will work only for products that have a class defined
            and the class has the associated attributes defined (qf_high, 
            qf_medium)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.grid.grid_type in  ['scan_angle', 'lonlatmesh']:
            coord1, coord2 = 'x', 'y'
        elif self.grid.grid_type == 'lonlat':
            coord1, coord2 = 'lon', 'lat'
        else:
            assert(False), f'{self.grid.grid_type} geht nich'
        if the_new_way:
            if data_quality == 'high':
                ds = self.grid.data_by_quality_high
            elif data_quality == 'high_medium':
                ds = self.grid.data_by_quality_high_medium
            elif data_quality == 'high_medium_low':
                ds = self.grid.data_by_quality_high_medium_low
            elif data_quality == 'medium':
                ds = self.grid.data_by_quality_medium
            elif data_quality == 'low':
                ds = self.grid.data_by_quality_low
            else:
                assert(False), f'data_quality not recognized: {data_quality}'
                
            # var_sel = [var for var in ds.variables if ds[var].dims == ('y', 'x')]
            var_sel = self.grid.valid_2D_variables
            # var_sel = list(ds.variables)
            # ds = self.grid.ds[self.grid.valid_2D_variables]
            
        else:
            if hasattr(self.grid, "valid_qf"):
                valid_qf = self.grid.valid_qf
                data_quality = 'custom'
            else:
                assert(hasattr(self.grid, 'qf_high')), 'Make sure the satellite product instance has the attributes qf_high and qf_medium defined.'
                if data_quality == 'high':
                    valid_qf = self.grid.qf_high
                elif data_quality == 'medium':
                    valid_qf = self.grid.qf_medium + self.grid.qf_high
                else:
                    assert(False), f'data_quality "{data_quality}" not in [good, intermediate].'
            
            
            
            # select relevant variablese ... those with x and y
            
            var_sel = [var for var in self.grid.ds.variables if self.grid.ds[var].dims == ('y', 'x')]
            
            # lat and lon might be present if lonlat was executed prior to this ... remove it
            if 'lon' in var_sel:
                var_sel.pop(var_sel.index('lon'))
                var_sel.pop(var_sel.index('lat'))
                        
            ds = self.grid.ds[var_sel]
            # cleanup the  coordinates
            coords2del = list(ds.coords)
            coords2del.remove('x')
            coords2del.remove('y')
            ds = ds.drop_vars(coords2del)
            
            # select only those values where QF is valid, only works if valid_qf was set.
            # assert(not isinstance(self.grid.valid_qf, type(None))), 'valid_qf can not be None, the file could probably not be assigned to a particular satellite product!!'
            # if not isinstance(self.grid.valid_qf, type(None)):
            ds = ds.where(ds.DQF.isin(valid_qf))
        for e,radius in enumerate(self.radii):
            wheres = self.distance_grids < radius
        
            ds_sel = ds.where(wheres)
            
            if radius == 10:
                self.tp_ds_sel = ds_sel.copy()
            # median
            # dst = ds_sel.median(dim = ['x', 'y'])
            dst = ds_sel.median(dim = [coord1, coord2])
            dst = dst.expand_dims({'stats':['median']})
        
            ds_area = dst
        
            #mean
            dst = ds_sel.mean(dim = [coord1, coord2])
            dst = dst.expand_dims({'stats':['mean']})
        
            ds_area = _xr.concat([ds_area, dst], 'stats')
        
            # std ... "un"-biased
            dst = ds_sel.std(dim = [coord1, coord2], 
                       ddof=1,
                      )
            dst = dst.expand_dims({'stats':['std']})
            ds_area = _xr.concat([ds_area, dst], 'stats')
        
            # no of valid points
            ds_area['num_of_valid_points'] = wheres.sum(dim = [coord1, coord2])
            # ds_area['num_of_valid_points'] = where.sum(dim = ['x','y'])
            
            ds_area = ds_area.expand_dims({'radius': [radius]})
            if e == 0:
                ds_area_all = ds_area
            else:
                ds_area_all = _xr.concat([ds_area_all, ds_area], 'radius')
        
        #### populate attributes
        for var in var_sel:
            try:
                ds_area_all[var].attrs = ds[var].attrs
            except KeyError:
                continue
            # ds_area_all[var].attrs['data_quality_assessment'] = data_quality
            # ds_area_all[var].attrs['data_quality_valid_qfs'] = valid_qf
            
        ds_area_all.radius.attrs['long_name'] = 'Radius of area around point over which data statistics are calsulated' 
        ds_area_all.stats.attrs['long_name'] = 'Statistics of values in circlular area around site'
        ds_area_all.num_of_valid_points.attrs['long_name'] = 'Number of valid data points used to calculating statistic.'
        
        if not the_new_way:
            if data_quality != 'custom':
                ds_area_all = ds_area_all.expand_dims({'data_quality':[data_quality,]})
                ds_area_all.data_quality.attrs[f'valid_qf_{data_quality}'] = valid_qf
        ds_area_all = ds_area_all.assign_coords({'data_quality': [data_quality,]})
        return ds_area_all
       
    @property
    def projection2area(self):
        if isinstance(self._projection2area, type(None)):
            if hasattr(self.grid, "valid_qf"):
                self._projection2area = self.projection2area('custom')
            else:
                out = []
                out.append(self._project2area('high'))
                try:
                    out.append(self._project2area('high_medium'))
                except KeyError:
                    pass
                try:
                    out.append(self._project2area('high_medium_low') )
                except KeyError:  
                    pass
                try:
                    out.append(self._project2area('medium'))
                except KeyError:
                    pass
                try:
                    out.append(self._project2area('low'))
                except KeyError:
                    pass
                out = _xr.concat(out, dim = 'data_quality', combine_attrs='drop_conflicts')
                self._projection2area = out
        return self._projection2area
    
    @property
    def deprecated_projection2area(self):
        if isinstance(self._projection2area, type(None)):

            # select relevant variablese ... those with x and y
            var_sel = [var for var in self.grid.ds.variables if self.grid.ds[var].dims == ('y', 'x')]
            
            var_sel.pop(var_sel.index('lon'))
            var_sel.pop(var_sel.index('lat'))
            
            ds = self.grid.ds[var_sel]
            
            # cleanup the the coordinates
            coords2del = list(ds.coords)
            coords2del.remove('x')
            coords2del.remove('y')
            ds = ds.drop_vars(coords2del)
            
            # select only those values where QF is valid, only works if valid_qf was set.
            # assert(not isinstance(self.grid.valid_qf, type(None))), 'valid_qf can not be None, the file could probably not be assigned to a particular satellite product!!'
            if not isinstance(self.grid.valid_qf, type(None)):
                ds = ds.where(ds.DQF.isin(self.grid.valid_qf))
            for e,radius in enumerate(self.radii):
                where = self.distance_grids < radius
            
                ds_sel = ds.where(where)
            
                # median
                dst = ds_sel.median(dim = ['x', 'y'])
                dst = dst.expand_dims({'stats':['median']})
            
                ds_area = dst
            
                #mean
                dst = ds_sel.mean(dim = ['x', 'y'])
                dst = dst.expand_dims({'stats':['mean']})
            
                ds_area = _xr.concat([ds_area, dst], 'stats')
            
                # std ... "un"-biased
                dst = ds_sel.std(dim = ['x', 'y'], 
                           ddof=1,
                          )
                dst = dst.expand_dims({'stats':['std']})
                ds_area = _xr.concat([ds_area, dst], 'stats')
            
                # no of valid points
                ds_area['num_of_valid_points'] = where.sum(dim = ['x','y'])
                ds_area['num_of_valid_points'] = where.sum(dim = ['x','y'])
                
                ds_area = ds_area.expand_dims({'radius': [radius]})
                if e == 0:
                    ds_area_all = ds_area
                else:
                    ds_area_all = _xr.concat([ds_area_all, ds_area], 'radius')
            
            #### populate attributes
            for var in var_sel:
                ds_area_all[var].attrs = ds[var].attrs
            ds_area_all.radius.attrs['long_name'] = 'Radius of area around point over which data statistics are calsulated' 
            ds_area_all.stats.attrs['long_name'] = 'Statistics of values in circlular area around site'
            ds_area_all.num_of_valid_points.attrs['long_name'] = 'Number of valid data points used to calculating statistic.'
            
            self._projection2area = ds_area_all
        return self._projection2area
            
    @property
    def distance_grids(self):
        if isinstance(self._distance_grids, type(None)):
            self.closest_grid_points
        return self._distance_grids
    
    @property
    def closest_grid_points(self):#, discard_outsid_grid = 2.2):
        if isinstance(self._closest_points, type(None)):
            #### TODO i think the block below showed be handled through a setter for the self.site attribute
            lon_lat_sites = self.sites
            #### TODO list of dicts needs to be done
            # print(f'lon_lat_sites type: {type(lon_lat_sites)}')
            # print(lon_lat_sites)
            if type(lon_lat_sites).__name__ == 'Station':
                idx = [lon_lat_sites.abb]
                lon_lat_sites = _np.array([[lon_lat_sites.lon, lon_lat_sites.lat]])
            elif type(lon_lat_sites).__name__ == 'NetworkStations':
                idx = [s.abb for s in lon_lat_sites]
                lon_lat_sites =_np.array([[s.lon, s.lat] for s in lon_lat_sites])
            elif isinstance(lon_lat_sites, type(None)):
                raise TypeError('It looks like no sites are defined to do the projection on.')
            elif isinstance(lon_lat_sites, dict):
                idx = [lon_lat_sites['abb']]
                lon_lat_sites = _np.array([[lon_lat_sites['lon'], lon_lat_sites['lat']]])
            elif isinstance(lon_lat_sites, list):
                idx = [s['abb'] for s in lon_lat_sites]
                lon_lat_sites = _np.array([[s['lon'], s['lat']] for s in lon_lat_sites])
            else:
                idx = range(len(lon_lat_sites))
            lon_g, lat_g = self.grid.lonlat
            # armins columns: argmin_x, argmin_y, lon_g, lat_g, lon_s, lat_s, dist_min
            # out_dict = {}
            
        #     if len(lon_g.shape) == 3:
        #         lon_g = lon_g[0,:,:]
        #         lat_g = lat_g[0,:,:]
            # index = idx #todo: rename below
            # for e,site in enumerate(lon_lat_sites):
    
            #     lon_s, lat_s = site
        
            #     p = _np.pi / 180
            #     a = 0.5 - _np.cos((lat_s-lat_g)*p)/2 + _np.cos(lat_g*p) * _np.cos(lat_s*p) * (1-_np.cos((lon_s-lon_g)*p))/2
            #     dist = 12742 * _np.arcsin(_np.sqrt(a))
        
            #     # get closest
            #     argmin = dist.argmin()//dist.shape[1], dist.argmin()%dist.shape[1]
            #     out[e,:2] = argmin        
            #     out[e,2] = lon_g[argmin]
            #     out[e,3] = lat_g[argmin]
            #     out[e,4] = lon_s
            #     out[e,5] = lat_s
            #     out[e,6] = dist[argmin]
            out, dist_array = get_dists(self.grid.lonlat, lon_lat_sites)
            # return out
            self.tp_out = (out, dist_array)
            if self.grid.grid_type in  ['scan_angle', 'lonlatmesh']:
                dist_array = _xr.DataArray(data = dist_array,
                                         dims = ['y','x','site'],
                                         coords = {'site': idx,
                                                   'x': self.grid.ds.x,
                                                   'y': self.grid.ds.y}
                                        )
                
            elif self.grid.grid_type == 'lonlat':
                assert('lon' in self.grid.ds.coords), 'arrrrg, not possible'
                dist_array = _xr.DataArray(data = dist_array,
                                         dims = ['lat','lon','site'],
                                         coords = {'site': idx,
                                                   'lon': self.grid.ds.lon,
                                                   'lat': self.grid.ds.lat}
                                        )
            else:
                assert(False), f'nenen {self.grid.grid_type} geht nich'
            self._distance_grids = dist_array
            self._closest_points = _pd.DataFrame(out, columns = ['argmin_x', 'argmin_y','lon_gritpoint', 'lat_gridpoint', 'lon_station', 'lat_station', 'distance_station_gridpoint'], index = idx)
            # closest_point = closest_point[closest_point.distance_station_gridpoint < discard_outsid_grid]
            # out_dict['closest_point'] = closest_point
            # out_dict['last_distance_grid'] = dist.astype(_np.float32)
        return self._closest_points

class QfManagment(object):
    def __init__(self, satellite_instance, qf_representation = 'as_is', qf_by_variable = None, global_qf = None, number_of_bits = None):
        """
        

        Parameters
        ----------
        satellite_instance : TYPE
            DESCRIPTION.
        qf_representation : str, optional
            How the DQF values ought to be represented, "as is" or "binary". The default is 'as_is'.
        qf_by_variable : TYPE, optional
            DESCRIPTION. The default is None.
        global_qf : TYPE, optional
            DESCRIPTION. The default is None.
        number_of_bits : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.satellite_instance = satellite_instance
        if qf_representation == 'binary':
            assert(isinstance(number_of_bits, int)), 'If qf_representation is "binary" the number_of_bits kwarg has to be set (integer).'
            self.qf_by_variable_binary = qf_by_variable
            self._qf_by_variable = None
        else:
            self._qf_by_variable = qf_by_variable
        self.qf_representation = qf_representation
        self.number_of_bits = number_of_bits
        self.global_qf = global_qf
    
    @property
    def qf_by_variable(self):
        if isinstance(self._qf_by_variable, type(None)):
            assert(not isinstance(self.qf_representation, type(None))), 'Either  qf_by_variable is not set or something is strange'
            self._qf_by_variable = self.get_assesment_mask()
        return self._qf_by_variable
    
    @qf_by_variable.setter
    def qf_by_variable(self, value):
        self._qf_by_variable = value
    
    def get_assesment_mask(self):
        def asses_qf(quality_flag, width, criteria):
            # print(f'type of criteria: {type(criteria)} ... {criteria}')
            # width = 8
            # criteria = [smoke_qf, global_qf]
            
            #### FIXME below I set forceobj to avoid the fallback warning. It would be nice to use nopython= True which causese an error :-(
            @vectorize([int8(float32)], target = 'cpu', forceobj = True)
            def _asses_qf(qf):
                if _np.isnan(qf):
                    return 0
                else:
                    qf = _np.int16(qf)
                    
                bins = _np.binary_repr(qf, width = width)[::-1]

                asses_list = []
                qf_list = []
                for cr in criteria:
                    if isinstance(cr, dict):
                        cr = [cr,]
                    qf_list+= cr
        
                for e,sqf in enumerate(qf_list):
                    for k in sqf:
                        sqfitem = sqf[k]
                        k = {'bad': 0, 'low':1, 'medium':2, 'high':3}[k]
        
                        b = ''
                        for pos in sqfitem['bins']:
                            b+=bins[pos]
                        b = b[::-1] # since we reversed the sting we have to revers it back ... sorry :-)
        
                        if int(b,2) in sqfitem['values']:
                            asses_list.append(k)
                asses_list.sort()
                return asses_list[0]
            return zip(quality_flag,_asses_qf(quality_flag))
        
        if self.qf_representation == 'binary':
            gfbyvarible = {}
            ds = self.satellite_instance.ds
            if isinstance(self.qf_by_variable_binary, type(None)):
                variables = self.satellite_instance.valid_2D_variables
            else:
                variables = self.qf_by_variable_binary
            qfdicts = []
            for var in variables:
                if not isinstance(self.qf_by_variable_binary, type(None)):
                    if self.qf_by_variable_binary[var] == 'ignore':
                        gfbyvarible[var] = 'ignore'
                        continue
                    qfdicts += [self.qf_by_variable_binary[var],]
                if not isinstance(self.global_qf, type(None)):
                    qfdicts += self.global_qf
                assert(len(qfdicts)>0), "this should not be possible"
                qf_asses_match = asses_qf(_np.unique(ds.DQF),self.number_of_bits, qfdicts)
                qf_asses_match = list(qf_asses_match)
        
                #### seperate into 
                gfdict = {}
                gfdict['bad'] = [qam[0] for qam in qf_asses_match if qam[1] == 0 ]
                gfdict['low'] = [qam[0] for qam in qf_asses_match if qam[1] == 1 ]
                gfdict['medium'] = [qam[0] for qam in qf_asses_match if qam[1] == 2 ]
                gfdict['high'] = [qam[0] for qam in qf_asses_match if qam[1] == 3 ]
                gfbyvarible[var] = gfdict
                # ds.Smoke # all
                # ds.Smoke.where(ds.DQF.isin(qf_high))
                # ds.Smoke.where(ds.DQF.isin(qf_high + qf_medium))
                # ds.Smoke.where(ds.DQF.isin(qf_high + qf_medium + qf_low))
        elif self.qf_representation == 'as_is':
            assert(len(self.global_qf) == 1), "programming qurired, iterate over the additional list items"
            gfbyvarible = {}
            for var in self.satellite_instance.valid_2D_variables:
                gfbyvarible[var] = self.global_qf[0]
        else:
            assert(False)
            
        return gfbyvarible

            

####---------------------------
#### Products classes
class ABI_L2_MCMIPC_M6(GeosSatteliteProducts):
    def __init__(self, *args):
        super().__init__(*args)
#         self._varname4test = 'CMI_C02'
        self._rgb = None

    @property
    def rgb(self):
        if isinstance(self._rgb, type(None)):
            self.generate_rgb()
        return self._rgb

    def generate_rgb(self, 
                        gamma = 1.8,#2.2, 
                        contrast = 130, #105
                       ):

        # get the rgb including the conversion from nIR to green
        
        channels_rgb = dict(red = self.ds['CMI_C02'].copy(),
                            green = self.ds['CMI_C03'].copy(),
                            blue = self.ds['CMI_C01'].copy())
        
        channels_rgb['green_true'] = 0.45 * channels_rgb['red'] + 0.1 * channels_rgb['green'] + 0.45 * channels_rgb['blue']
        
        # adjust image, e.g gamma etc
        for chan in channels_rgb:
            col = channels_rgb[chan]
            # Apply range limits for each channel. RGB values must be between 0 and 1
            try:
                new_col = col / col.max()
            except ValueError:
                print('No valid data in at least on of the channels')
                assert(False)
                # return False
            
            # apply gamma
            if not isinstance(gamma, type(None)):
                new_col = new_col**(1/gamma)
            
            # contrast
            #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
            if not isinstance(contrast, type(None)):
                cfact = (259*(contrast + 255))/(255.*259-contrast)
                new_col = cfact*(new_col-.5)+.5
            
            channels_rgb[chan] = new_col
        
        # shape for plotting
        red = channels_rgb["red"].copy()
        green = channels_rgb["green_true"].copy()
        blue = channels_rgb["blue"].copy()
        
        # Stack them along the last axis to create an RGB image
        image_data = _np.stack([red, green, blue], axis=-1)
        image_data = _np.clip(image_data,0,1)
        # self.tp_image = image_data
        # color = image_data.reshape(-1, 3)
        coords = {'x': self.ds.x, 'y': self.ds.y, 'rgb':['r','g','b']}
        da = _xr.DataArray(image_data, dims= ['y', 'x', 'rgb'], coords=coords)
        self.ds['true_color'] = da
        self._rgb = da
        return da

    def plot_true_color(self, **kwargs):
        out = {}
        self.rgb
        bmap,pc,cb = self.plot('true_color', 
                        # color = self.rgb,
                        colorbar = False,
                        linewidth = 0,
                        **kwargs)
        # pc.set_linewidths(0.001)
        # pc.set_edgecolor([0,0,0,0])
        out['pc'] = pc
#             plt.title('GOES-16 True Color', loc='left', fontweight='semibold', fontsize=15)
#             plt.title('%s' % scan_start.strftime('%d %B %Y %H:%M UTC'), loc='right');
        out['bmap'] = bmap
        return out

#     def dep_plot_true_color(self, 
#                         gamma = 1.8,#2.2, 
#                         contrast = 130, #105
#                         projection = None,
#                         bmap = None,
#                         width = 5e6,
#                         height = 3e6,
#                         **kwargs,
#                        ):
#         out = {}
#         channels_rgb = dict(red = self.ds['CMI_C02'].data.copy(),
#                             green = self.ds['CMI_C03'].data.copy(),
#                             blue = self.ds['CMI_C01'].data.copy())
        
#         channels_rgb['green_true'] = 0.45 * channels_rgb['red'] + 0.1 * channels_rgb['green'] + 0.45 * channels_rgb['blue']

        
        
#         for chan in channels_rgb:
#             col = channels_rgb[chan]
#             # Apply range limits for each channel. RGB values must be between 0 and 1
#             try:
#                 new_col = col / col[~_np.isnan(col)].max()
#             except ValueError:
#                 print('No valid data in at least on of the channels')
#                 return False
            
#             # apply gamma
#             if not isinstance(gamma, type(None)):
#                 new_col = new_col**(1/gamma)
            
#             # contrast
#             #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
#             if not isinstance(contrast, type(None)):
#                 cfact = (259*(contrast + 255))/(255.*259-contrast)
#                 new_col = cfact*(new_col-.5)+.5
            
#             channels_rgb[chan] = new_col
        
#         rgb_image = _np.dstack([channels_rgb['red'],
#                              channels_rgb['green_true'],
#                              channels_rgb['blue']])
#         rgb_image = _np.clip(rgb_image,0,1)
#         self.tp_rgb = rgb_image
        
#         a = _plt.subplot()
#         if isinstance(projection, type(None)) and isinstance(bmap, type(None)):
            
#             a.imshow(rgb_image)
# #             a.set_title('GOES-16 RGB True Color', fontweight='semibold', loc='left', fontsize=12);
# #             a.set_title('%s' % scan_start.strftime('%d %B %Y %H:%M UTC '), loc='right');
#             a.axis('off')
    
#         else:          
#             lons,lats = self.lonlat
            
#             # Make a new map object Lambert Conformal projection
#             if not isinstance(bmap,_Basemap):
#                 bmap = _Basemap(resolution='i', projection='aea', area_thresh=5000, 
#                              width=width, height=height, 
#     #                          lat_1=38.5, lat_2=38.5, 
#                              lat_0=38.5, lon_0=-97.5)

#                 bmap.drawcoastlines()
#                 bmap.drawcountries()
#                 bmap.drawstates()

#             # Create a color tuple for pcolormesh

#             # Don't use the last column of the RGB array or else the image will be scrambled!
#             # This is the strange nature of pcolormesh.
#             rgb_image = rgb_image[:,:-1,:]

#             # Flatten the array, becuase that's what pcolormesh wants.
#             colortuple = rgb_image.reshape((rgb_image.shape[0] * rgb_image.shape[1]), 3)

#             # Adding an alpha channel will plot faster, according to Stack Overflow. Not sure why.
#             colortuple = _np.insert(colortuple, 3, 1.0, axis=1)

#             # We need an array the shape of the data, so use R. The color of each pixel will be set by color=colorTuple.
#             pc = bmap.pcolormesh(lons, lats, channels_rgb['red'], color=colortuple, linewidth=0, latlon=True, **kwargs)
#             pc.set_array(None) # Without this line the RGB colorTuple is ignored and only R is plotted.
#             out['pc'] = pc
# #             plt.title('GOES-16 True Color', loc='left', fontweight='semibold', fontsize=15)
# #             plt.title('%s' % scan_start.strftime('%d %B %Y %H:%M UTC'), loc='right');
#             out['bmap'] = bmap
#         return out

# class ABI_L2_AOD(GeosSatteliteProducts):
#     def __init__(self, *args):
#         super().__init__(*args)
#         self.valid_qf = [0,1]
        
# class ABI_L2_LST(GeosSatteliteProducts):
#     def __init__(self, *args):
#         super().__init__(*args)
#         self.valid_qf = [0,]
        
# class ABI_L2_ACHA(GeosSatteliteProducts):
#     def __init__(self, *args):
#         '''Cloud Top Height'''
#         super().__init__(*args)
#         # self.valid_qf = [0,]    
        
# class ABI_L2_COD(GeosSatteliteProducts):
#     def __init__(self, *args):
#         '''Cloud Optical Depth'''
#         super().__init__(*args)
#         self.valid_qf = [0,]  
   
#######################################
#### Below use assesment dataset
#################################  
class GoesExceptionVerionNotRecognized(Exception):
    def __init__(self,message = 'Product version not recognized'):
        # txt = f"The version of this product ({si.product_info['version']}) has not been tested and might return false results."#"\nfullname: {si.product_name}"
        # if not isinstance(message, type(None)):
        #     txt +='\n'+message
        self.message = message
        super().__init__(self.message)

class ABI_L2_LST(GeosSatteliteProducts):
    def __init__(self, *args):
        super().__init__(*args)
        
        #quality flags changed at some point
        
        if self.product_info['version'] in ['M3',]:
            print('bubasd')
            global_qf = [{'high':   [0], 
                          'medium': [8],
                          # 'low':    [2],
                          'bad':    [2,4,16,32]}]
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
        elif self.product_info['version'] in ['M6',]:
            global_qf = [{'high':   [0], 
                          'medium': [1],
                           'low':   [2],
                          'bad':    [3]}]
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
        else:
            if 'PQI' in self.ds.variables:
                message = 'PQI in variables -> qf probably similar to M6'
            else:
                message = 'PQI in variables -> qf probably similar to M3'
            raise GoesExceptionVerionNotRecognized(self, message)
            
            
class ABI_L2_AOD(GeosSatteliteProducts):
    def __init__(self, *args):
        super().__init__(*args)
        # self.valid_qf = [0,1]
        
        if self.ds.DQF.attrs['flag_meanings'] == 'high_quality_retrieval_qf medium_quality_retrieval_qf low_quality_retrieval_qf no_retrieval_qf':
        # if self.product_info['version'] in ['M6',]:
            
            global_qf = [{'high':   [0], 
                          'medium': [1],
                          'low':    [2],
                          'bad':    [3]}]
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
        elif self.ds.DQF.attrs['flag_meanings'] == 'good_retrieval_qf bad_retrieval_qf':
        # elif self.product_info['version'] in ['M3',]:
            global_qf = [{'high':   [0], 
                          'low': [1],
                          }]
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
        else:
            raise GoesExceptionVerionNotRecognized(self)

class ABI_L2_COD(GeosSatteliteProducts):
    def __init__(self, *args, night = False):
        '''Cloud Optical Depth'''
        super().__init__(*args)
        # self.valid_qf = [0,]  
        
        if self.product_info['version'] in ['M3','M6',]:
            if night:
                qf0bad = 0
            else:
                qf0bad = 1 
            global_qf = [{'bad':   {'bins': [0], 'values': [qf0bad,]}}, 
                         {'high':   {'bins': [1,2,3,4], 'values': [0]}},  
                         {'medium': {'bins': [1,2,3,4], 'values': [1,2,5,8]}},
                         {'bad': {'bins': [1,2,3,4], 'values': [3,4,6,7]}},
                        ]
            self.qf_managment = QfManagment(self, qf_representation='binary', global_qf=global_qf, number_of_bits=5)
            
        else:
            raise GoesExceptionVerionNotRecognized(self)

class ABI_L2_ACM(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Clear Sky Mask'''
        super().__init__(*args)
        
        # self.qf_high = [0,]
        # self.qf_medium = [2,4,5,6]
        # self.qf_low = None
        # self.qf_bad = [1,3]
        
        if self.product_info['version'] in ['M3','M6',]:
            global_qf = [{'high':   [0], 
                          'medium': [2,4,5,6],
                          'bad':    [1,3]}]
            
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
            
        else:
            raise GoesExceptionVerionNotRecognized(self)
        
class ABI_L2_ADP(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Clear Sky Mask'''
        super().__init__(*args)
        # self.qf_representation = 'binary'
        
        if self.product_info['version'] in ['M6',]:
            qf_by_variable =  {'Aerosol': 'ignore',
                               'Smoke': {"bad":    {'bins': [0,],  'values':[1,]},
                                         "low" :   {'bins': [2,3], 'values':[0,]},
                                         "medium": {'bins': [2,3], 'values':[1,]},
                                         "high":   {'bins': [2,3], 'values':[3,]},},
                                'Dust': {"bad":    {'bins': [1,],  'values':[1,]},
                                         "low" :   {'bins': [4,5], 'values':[0,]},
                                         "medium": {'bins': [4,5], 'values':[1,]},
                                         "high":   {'bins': [4,5], 'values':[3,]},}}
            
            global_qf = [{'bad': {'bins': [6], 'values': [1]}}, {'bad': {'bins': [7], 'values': [1]}}]
            self.qf_managment = QfManagment(self, qf_representation='binary', qf_by_variable = qf_by_variable, global_qf= global_qf, number_of_bits=8)
            
        else:
            raise GoesExceptionVerionNotRecognized(self)
        
        
        
class ABI_L2_ACHA(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Cloud Top Height'''
        super().__init__(*args)
        if self.product_info['version'] in ['M6',]:
            global_qf = [{'high': [0], 'bad': [1,2,3,4,5,6]}]
            
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
            
        else:
            txt = f'Product version {self.product_info["version"]} unknown for product {{self.product_info["name"]}}.'
            raise GoesExceptionVerionNotRecognized(txt)
        
class ABI_L2_CTP(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Cloud Top Pressure'''
        super().__init__(*args)
        
        # if self.product_info['version'] in ['bla',]:
        if self.ds.DQF.attrs['flag_meanings'] == 'good_quality_qf invalid_due_to_not_geolocated_qf invalid_due_to_LZA_threshold_exceeded_qf invalid_due_to_bad_or_missing_brightness_temp_data_qf invalid_due_to_clear_or_probably_clear_sky_qf invalid_due_to_unknown_cloud_type_qf invalid_due_to_nonconvergent_retrieval_qf':
            global_qf = [{'high': [0], 'bad': [1,2,3,4,5,6]}]
            
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
            
        elif self.ds.DQF.attrs['flag_meanings'] == 'good_quality_qf marginal_quality_qf retrieval_attempted_qf bad_quality_qf opaque_retrieval_qf':
            global_qf = [{'high': [0], 'medium': [1], 'low': [2], 'bad': [3,4]}]
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            global_qf= global_qf, 
                                           )
            
        else:
            print(self.ds.t)
            raise GoesExceptionVerionNotRecognized(self)
            
            
class ABI_L2_DSR(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Downwelling Shortwave Radiation'''
        super().__init__(*args)

        if self.ds.DQF.attrs['flag_meanings'] == 'good_quality_qf degraded_quality_or_invalid_qf':
        # if self.product_info['version'] in ['bla',]:
        # if self.product_info['version'] in ['M6',]:

            global_qf = [{'high': [0], 'bad': [1,]}]
            
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            # qf_representation='binary', 
                                            # qf_by_variable = qf_by_variable, 
                                            global_qf= global_qf, 
                                            # number_of_bits=8
                                           )
            
        else:
            print(self.ds.t)
            raise GoesExceptionVerionNotRecognized(self)
            
class ABI_L2_SRB(GeosSatteliteProducts):
    def __init__(self, *args):
        '''Surface radiative budget. This is an experimental product'''
        super().__init__(*args)
        
        if self.product_info['version'] in ['bla',]:
            global_qf = [{'high': [0], 'bad': [1,]}]
            
            self.qf_managment = QfManagment(self, 
                                            qf_representation='as_is', 
                                            # qf_representation='binary', 
                                            # qf_by_variable = qf_by_variable, 
                                            global_qf= global_qf, 
                                            # number_of_bits=8
                                           )
            
        else:
            raise GoesExceptionVerionNotRecognized(self)
        
        
############################################
#### Below are the JPSS products

class JRR_AOD(GeosSatteliteProducts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.valid_qf = [0,1]
        
        if self.product_info['version'] in [3.0,]:
            
            global_qf = [{'high':   [0], 
                          'medium': [1],
                          'low':    [2],
                          'bad':    [3]}]
            self.qf_managment = QfManagment(self, 
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
            raise GoesExceptionVerionNotRecognized(self)


############################################
#### specialized function ... probably of limited usefullness for the average user    
def projection_function(row, stations, test = False, verbose = False):
    """
    This function is used for on the fly processing (projection to site) for the nedis_aws package.
    The function allows for projection while downloading and subsequent discarding 
    of satellite data 

    Parameters
    ----------
    row : TYPE
        DESCRIPTION.
    stations : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # read the file
    ngsinst = open_file(row.path2file_local)
    
    # project to stations
    projection = ngsinst.project_on_sites(stations)
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
        if f'{var}_DQF_assessed' in point.variables:
            point = point.rename({f'{var}_DQF_assessed': f'{var}_on_pixel_DQF_assessed',})
    point = point.rename({'DQF': 'DQF_on_pixel'})
    if test:
        return projection
    # merge aerea and point
    ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
    # add a time stamp
    dt = _pd.Series([_pd.to_datetime(ngsinst.ds.attrs['time_coverage_start']), _pd.to_datetime(ngsinst.ds.attrs['time_coverage_end'])]).mean().to_datetime64()
    ds = ds.expand_dims({'datetime': [dt]}, )
    
    # there was another time coordinate without dimention though ... dropit
    ds = ds.drop_vars('t')

    # global attribute
    ds.attrs['info'] = ('This file contains a projection of satellite data onto SURFRAD sites.\n'
                         'It includes the closest pixel data as well as the average over circular\n'
                         'areas with various radii. Note, for the averaged data only data is\n'
                         'considered with a qulity flag given by the prooduct class in the\n'
                         'nesdis_gml_synergy package.')
    

    # save2file
    if test:
        return ds
    ds.to_netcdf(row.path2file_local_processed)
    # Memory kept on piling up -> maybe a cleanup will help
    ds.close()
    ds = None
    ngsinst.ds.close()
    ngsinst = None
    
    return 

def projection_function_multi(row, error_queue, stations = None, verbose = True):
    try:
        # if verbose:
        #     print('projection_function_multi')
    
        if row.path2file_local_processed.is_file():
            if verbose:
                print(f'file exists... skip {row.path2file_local_processed}')
            return
        if not row.path2file_local.is_file():
            print('^', end = '', flush = True)
            #### download        
            # self.aws.clear_instance_cache()     #-> not helping           
            aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True) #- not helping
            print('.', end = '', flush = True)
            aws.clear_instance_cache()
            print('.', end = '', flush = True)
            # print(f'aws.get({row.path2file_aws.as_posix()}, {row.path2file_local.as_posix()}')
            aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
            print('-', end = '', flush = True)
        #### process
        # return
        raise_exception = True
        try:
            rowold = row.copy()
            projection_function(row, stations)
            if not row.equals(rowold):
                print('row changed ... return')
                return row, rowold
            if verbose:
                print(':', end = '', flush = True)
        except :
            if raise_exception:
                raise
            else:
                print(f'error applying function on one file {row.path2file_local.name}. The raw fill will still be removed (unless keep_files is True) to avoid storage issues')
        #### remove raw file
        # if not self.keep_files:
        row.path2file_local.unlink()
        if verbose:
            print('|', end = '', flush = True)

    except Exception as e:
        error_queue.put(e)#traceback.format_exc())
    return

def projection_function_test(row, stations):
    """
    This function is used for on the fly processing (projection to site) for the nedis_aws package.
    The function allows for projection while downloading and subsequent discarding 
    of satellite data 

    Parameters
    ----------
    row : TYPE
        DESCRIPTION.
    stations : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # read the file
    ngsinst = open_file(row.path2file_local)
    
    # project to stations
    projection = ngsinst.project_on_sites(stations)

    # merge closest gridpoint and area
    point = projection.projection2point.copy()#.sel(site = 'TBL')
    point['DQF'] = point.DQF.astype(int) # for some reason this was float64... because there are some nans in there
    
    # change var names to distinguish from area
    for var in ngsinst.valid_2D_variables:
        point = point.rename({var: f'{var}_on_pixel',})
        point = point.rename({f'{var}_DQF_assessed': f'{var}_on_pixel_DQF_assessed',})
    point = point.rename({'DQF': 'DQF_on_pixel'})
    
    # merge aerea and point
    ds = projection.projection2area.merge(point)#.rename({alt_var: f'{alt_var}_on_pixel', 'DQF': 'DQF_on_pixel'}))
    
    # add a time stamp
    dt = _pd.Series([_pd.to_datetime(ngsinst.ds.attrs['time_coverage_start']), _pd.to_datetime(ngsinst.ds.attrs['time_coverage_end'])]).mean().to_datetime64()
    ds = ds.expand_dims({'datetime': [dt]}, )
    
    # there was another time coordinate without dimention though ... dropit
    ds = ds.drop_vars('t')

    # global attribute
    ds.attrs['info'] = ('This file contains a projection of satellite data onto SURFRAD sites.\n'
                         'It includes the closest pixel data as well as the average over circular\n'
                         'areas with various radii. Note, for the averaged data only data is\n'
                         'considered with a qulity flag given by the prooduct class in the\n'
                         'nesdis_gml_synergy package.')
    

    # save2file
    ds.to_netcdf(row.path2file_local_processed)
    # Memory kept on piling up -> maybe a cleanup will help
    ds.close()
    ds = None
    ngsinst.ds.close()
    ngsinst = None
    
    return 

########################################################
#### Is this still used?
abi_products = [{'product_name': 'ABI-L2-ACHA',
                'long_name': 'Cloud Top Height', 
                'satlab_class': ABI_L2_ACHA},
                # {'product_name': 'ABI-L2-ACHT',
                # 'long_name': 'Cloud Top Temperature', 
                # 'satlab_class': ABI_L2_ACHT},
               ]


##########################################################
##### temp files for jpssscraper develop
