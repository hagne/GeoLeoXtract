#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:01:13 2023

@author: hagen

Known issues: 
    mapping edge issue in longiturdinal direction (only for leo satellites):
        If a site is very close to the edge with respect to its longitude 
        direction, the area could touch the edge of the scann area and create a 
        bias. The same is not happening for the Latitude as the cloud_interface 
        takes the two closest scannes in that direction.
"""

# -*- coding: utf-8 -*-
import pathlib as _pl
import pandas as _pd
import s3fs as _s3fs
# import urllib as _urllib
# import html2text as _html2text
import psutil as _psutil
import numpy as _np
# import xarray as _xr
import warnings
from functools import partial
import multiprocessing as mp
import nesdis_gml_synergy.info as ngsinfo

def readme():
    url = 'https://docs.opendata.aws/noaa-goes16/cics-readme.html'
    # html = _urllib.request.urlopen(url).read().decode("utf-8") 
    # out = _html2text.html2text(html)
    # print(out)
    print(f'follow link for readme: {url}')
    
# moved to info module
# variable_info = {'ABI-L1b-Rad': 'Radiances',
#                  'ABI-L2-ACHA': 'Cloud Top Height',
#                  'ABI-L2-ACHA2KM': 'Cloud Top Height 2km res. (todo: verify!!)',
#                  'ABI-L2-ACHP2KM': 'Cloud Top Pressure 2km res. (todo: verify)',
#                  'ABI-L2-ACHT': 'Cloud Top Temperature',
#                  'ABI-L2-ACM': 'Clear Sky Mask',
#                  'ABI-L2-ACTP': 'Cloud Top Phase',
#                  'ABI-L2-ADP': 'Aerosol Detection',
#                  'ABI-L2-AICE': 'Ice Concentration and Extent',
#                  'ABI-L2-AITA': 'Ice Age and Thickness',
#                  'ABI-L2-AOD': 'Aerosol Optical Depth',
#                  'ABI-L2-BRF': 'Land Surface Bidirectional Reflectance Factor () 2 km resolution & DQFs',
#                  # 'ABI-L2-CCL': 'unknown',
#                  'ABI-L2-CMIP': 'Cloud and Moisture Imagery',
#                  'ABI-L2-COD': 'Cloud Optical Depth',
#                  'ABI-L2-COD2KM': 'Cloud Optical Depth 2km res.',
#                  'ABI-L2-CPS': 'Cloud Particle Size',
#                  'ABI-L2-CTP': 'Cloud Top Pressure',
#                  'ABI-L2-DMW': 'Derived Motion Winds',
#                  'ABI-L2-DMWV': 'L2+ Derived Motion Winds',
#                  'ABI-L2-DSI': 'Derived Stability Indices',
#                  'ABI-L2-DSR': 'Downward Shortwave Radiation',
#                  'ABI-L2-FDC': 'Fire (Hot Spot Characterization)',
#                  'ABI-L2-LSA': 'Land Surface Albedo () 2km resolution & DQFs',
#                  'ABI-L2-LST': 'Land Surface Temperature',
#                  'ABI-L2-LST2KM': 'Land Surface Temperature',
#                  'ABI-L2-LVMP': 'Legacy Vertical Moisture Profile',
#                  'ABI-L2-LVTP': 'Legacy Vertical Temperature Profile',
#                  'ABI-L2-MCMIP': 'Cloud and Moisture Imagery',
#                  'ABI-L2-RRQPE': 'Rainfall Rate (Quantitative Precipitation Estimate)',
#                  'ABI-L2-RSR': 'Reflected Shortwave Radiation Top-Of-Atmosphere',
#                  'ABI-L2-SST': 'Sea Surface (Skin) Temperature',
#                  'ABI-L2-TPW': 'Total Precipitable Water',
#                  'ABI-L2-VAA': 'Volcanic Ash: Detection and Height'}

# satellite_list = [dict(name = 'NOAA 20', type_of_orbit = 'leo')]

def get_available_JPSS_products(sensor = 'VIIRS', verbose = False):
    """
    In deveolpment. All seems to be there, do it!!!

    Parameters
    ----------
    sensor : TYPE, optional
        DESCRIPTION. The default is 'VIIRS'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert(sensor == 'VIIRS'), 'only VIIRS available at this point, other sensors ar available on AWS but not yet implemented, do it!!'
    # satellites = ['NOAA20', 'SNPP']
    aws = _s3fs.S3FileSystem(anon=True)
    jpss_base_folder = _pl.Path('noaa-jpss')
    
    satellites_available = aws.glob(jpss_base_folder.joinpath('*').as_posix())
    exclude_sat_fld = ['index.html', 
                        'JPSS_Blended_Products',
                       ]
    satellites_available = [s.split('/')[-1] for s in satellites_available if s.split('/')[-1] not in exclude_sat_fld]
    # return satellites_available
    if verbose:
        print(f'satellites: {satellites_available}')
    products = []
    # for satellite in [16,17]:
    for satellite in satellites_available:
        # satellite = 16#16 (east) or 17(west)
        # base_folder = jpss_base_folder.joinpath(satellite)
        sensors_available = aws.glob(jpss_base_folder.joinpath(satellite).joinpath('*').as_posix())
        sensors_available = [s.split('/')[-1] for s in sensors_available]
        assert(sensor in sensors_available), f"The sensor {sensor} does not seem to be on the {satellite} satellite. Choose from {sensors_available}"
        products_available = aws.glob(jpss_base_folder.joinpath(satellite).joinpath(sensor).joinpath('*').as_posix())
        # df[satellite] = 
        products += [p.split('/')[-1].replace(f'{satellite}_', '') for p in products_available if '.pdf' not in p]
    products = _np.unique(products)
    # return products
    # products = _np.unique([p[:-1] for p in products])
    # products = [p for p in products if p[:3] =='ABI']
    product_avail = _pd.DataFrame(columns = satellites_available, index = products)
    # return product_avail
    def get_first_day(product, satellite, verbose = False):
        if verbose:
            print('----------------------')
            print(f'satellite: {satellite}')
        prod = f'{satellite}_{product}'
        # product_folder = _pl.Path(f'{jpss_base_folder}/{sensor}/{prod}')
        product_folder = _pl.Path(f'{jpss_base_folder}/{satellite}/{sensor}/{prod}')
        if verbose:
            print(f'product_folder: {product_folder}')
        years = aws.glob(product_folder.joinpath('*').as_posix())
        years = [y for y in years if '.txt' not in y]
        if verbose: 
            print(f'years: {years}')
        if len(years) == 0:
            return False

        years.sort()
        yearfolder = _pl.Path(years[0])
        if verbose:
            print(f'firstyear: {yearfolder.name}')
        # is2000 = True
        # while is2000:
        #     yearfolder = years.pop(0)
        #     firstyear = yearfolder.split('/')[-1]
        #     # print(firstyear)
        #     if firstyear != '2000':
        #         is2000 = False
        if verbose:
            print(f'yearfolder: {yearfolder}')
            
            
        months = aws.glob(_pl.Path(yearfolder).joinpath('*').as_posix())
        months.sort()
        if verbose:
            print(f'months: {months}')
        monthsfolder = _pl.Path(months[0])
        # fd = _pd.to_datetime(firstyear) + _pd.to_timedelta(firstday, "D")
        if verbose:
            print(f'firstmonth: {monthsfolder.name}')
            
            
        days = aws.glob(monthsfolder.joinpath('*').as_posix())
        days.sort()
        if verbose:
            print(f'days: {days}')
        dayfolder = _pl.Path(days[0])
        # firstday = days.nae
        
        fd_str = f'{yearfolder.name}-{monthsfolder.name}-{dayfolder.name}'
        fd = _pd.to_datetime(fd_str)
        if verbose:
            print(f'first day: {fd}')
        return fd

    for prod, row in product_avail.iterrows():
        if verbose:
            print('=====================')
            print(f'prod: {prod}')
            print('=====================')
            
        if 'Gridded' in prod:
            if verbose:
                print('gridded .... condintue')
            continue
        # if prod != 'VIIRS_Aerosol_Optical_Depth_EDR':
        #     continue

        for idx,val in row.items():
            # break
            # satellite, scan_sector = idx.split('-')
            satellite = idx
            fd = get_first_day(prod, satellite, verbose = verbose)
            if fd:
                product_avail.loc[prod, idx] = f'{fd.year:04d}-{fd.month:02d}-{fd.day:02d}'
            else:
                product_avail.loc[prod, idx] = '-'
        # break
                
    # product_avail.insert(0,'longname', product_avail.apply(lambda row: variable_info[row.name], axis=1))
    product_avail.index.name = 'product'
    return product_avail

def get_available_GOES_products(sensor = 'ABI', verbose = False):
    assert(sensor == 'ABI'), 'Only ABI sensors at this time'
    satellites = ['noaa-goes16', 'noaa-goes17']
    aws = _s3fs.S3FileSystem(anon=True)

    products = []
    # for satellite in [16,17]:
    for satellite in satellites:
        # satellite = 16#16 (east) or 17(west)
        base_folder = _pl.Path(satellite)
        products_available = aws.glob(base_folder.joinpath('*').as_posix())
        # df[satellite] = 
        products += [p.split('/')[-1] for p in products_available if '.pdf' not in p]
        
    products = _np.unique(products)
    products = _np.unique([p[:-1] for p in products])
    products = [p for p in products if p[:3] =='ABI']
    # product_avail = _pd.DataFrame(columns = ['16-C', '16-F', '16-M', '17-C', '17-F', '17-M'], index = products)
    product_avail = _pd.DataFrame(columns = ['16-C', '16-F', '16-M', '17-C', '17-F', '17-M', '18-C', '18-F', '18-M'], index = products)

    def get_first_day(product, satellite, scan_sector):
        prod = product
        product_folder = _pl.Path(f'noaa-goes{satellite}/{prod}{scan_sector}')

        years = aws.glob(product_folder.joinpath('*').as_posix())
        if len(years) == 0:
            return False

        years.sort()

        # there is this strange folder from 2000, no Idea what that is about?!?
        is2000 = True
        while is2000:
            yearfolder = years.pop(0)
            firstyear = yearfolder.split('/')[-1]
            # print(firstyear)
            if firstyear != '2000':
                is2000 = False

        yearfolder = _pl.Path(yearfolder)
        days = aws.glob(yearfolder.joinpath('*').as_posix())
        days.sort()
        firstday = int(days[0].split('/')[-1])
        fd = _pd.to_datetime(firstyear) + _pd.to_timedelta(firstday, "D")
        return fd
    
    # return  product_avail
    for prod, row in product_avail.iterrows():
        # break
        if verbose:
            print(f'product: {prod}', end = '')
        for idx,val in row.items():
            if verbose:
                print(f'{idx}, ', end = '')
            # break
            satellite, scan_sector = idx.split('-')
            fd = get_first_day(prod, satellite, scan_sector)
            if fd:
                product_avail.loc[prod, idx] = f'{fd.year:04d}-{fd.month:02d}-{fd.day:02d}'
            else:
                product_avail.loc[prod, idx] = '-'
        if verbose:
            print('.')
    
    
    def assign_variable_info(row):
        try:
            out = ngsinfo.VIIRS_product_info[row.name]
        except KeyError:
            out = 'N/A; add info to "variable_info"'
        return out
    
    product_avail.insert(0,'longname', product_avail.apply(assign_variable_info, axis=1))
    product_avail.index.name = 'product'
    return product_avail

get_available_products = get_available_GOES_products





class AwsQuery(object):
    def __init__(self,
                 path2folder_local = '/mnt/telg/tmp/aws_tmp/',
                 satellite = '16',               
                 sensor = 'VIIRS',
                 product = 'ABI-L2-AOD',
                 scan_sector = 'C',
                 start = '2020-08-08 20:00:00', 
                 end = '2020-08-09 18:00:00',
                 site = None,
                 process = None,
                 keep_files = None,
                 verbose = False,
                 overwrite = False,
                 # check_if_file_exist = True,
                 # no_of_days = None,
                 # last_x_days = None, 
                 # max_no_of_files = 100,#10*24*7,
                ):
        """
        This will initialize a search on AWS.

        Parameters
        ----------
        path2folder_local : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/tmp/aws_tmp/'.
        satellite : TYPE, optional
            DESCRIPTION. The default is '16'.
        product : str, optional
            Note this is the product name described at 
            https://docs.opendata.aws/noaa-goes16/cics-readme.html 
            but without the scan sector. The default is 'ABI-L2-AOD'.
        scan_sector : str, optional
            (C)onus, (F)ull_disk, (M)eso. The default is 'C'.
        start : TYPE, optional
            DESCRIPTION. The default is '2020-08-08 20:00:00'.
        end : TYPE, optional
            DESCRIPTION. The default is '2020-08-09 18:00:00'.
        process: dict,
            This is still in development and might be buggy.
            Example:
                dict(concatenate = 'daily',
                     function = lambda row: some_function(row, *args, **kwargs),
                     prefix = 'ABI_L2_AOD_processed',
                     path2processed = '/path2processed/')
        keep_files: bool, optional
            Default is True unless process is given which changes the default
            False.
        overwrite: bool, optional
            If existing files are removed from the workplan.

        Returns
        -------
        None.

        """
        self.satellite = satellite
        self.path2folder_aws = _pl.Path(f'noaa-goes{self.satellite}')
        
        self.sensor = sensor
        self.scan_sector = scan_sector 
        self.product = product
        
        self.start = _pd.to_datetime(start)
        self.end =  _pd.to_datetime(end)
        
        self.path2folder_local = _pl.Path(path2folder_local)
        self.overwrite = overwrite
        
        if isinstance(site, dict):
            self.site = [type('athoc_site', (), site),]
        else:
            if isinstance(site, list):
                self.site = site
            else:
                self.site = [site]
                
        
        sli = [s for s in ngsinfo.satellite_list if satellite in s['names']]
        if len(sli) == 0:
            self.type_of_orbit = 'geo'
            self.satellite_info = 'FIXIT'
        else:            
            self.satellite_info = sli[0]
            self.satellite = self.satellite_info['names'][0]
            assert(self.satellite == 'NOAA 20'), 'somewhere NOAA20 is hard coded!!!! find it and fix it!!'
            self.type_of_orbit = self.satellite_info['type_of_orbit']
        
        
        if isinstance(process, dict):
            self._process = True
            # self._process_concatenate = process['concatenate']
            self._process_function = process['function']
            self._process_name_prefix = process['prefix']
            self._process_path2processed = _pl.Path(process['path2processed'])
            # self._process_path2processed_tmp = self._process_path2processed.joinpath('tmp')
            # self._process_path2processed_tmp.mkdir(exist_ok=True)
            if keep_files:
                self.keep_files = True
            else:
                self.keep_files = False
            # self.check_if_file_exist = False
        else:
            self._process = False
        
        #### TODO memoryleak initiate aws only when its needed
        self.aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
        # self.aws.clear_instance_cache() # strange things happen if the is not the only query one is doing during a session
        # properties
        self._workplan = None
        self._overpassplan = None
        self._verbose = verbose
    
    def _overpasstime2filesincloud(self, row, raise_error = False):
        srow = row
        
        #### get files on aws for each overpass
        ### match product names to the product accronyme (sometime, like in case of AOD there are multiple possibilities
        sattt = self.satellite.replace(" ","")
        products_leo = [dict(abb = 'aod', full_names = ['Aerosol_Optical_Depth_EDR', 'Aerosol_Optical_Depth_EDR_Reprocessed'])]
        
        full_names = [p for p in products_leo if p['abb'] == self.product.lower()][0]['full_names']
        # full_names
        # fn = full_names[0]
        
        ### get all files 
        afiles = []
        # If there are multiple names only one should actually have files in them
        for fn in full_names:
            ponaws = f'noaa-jpss/{sattt}/{self.sensor}/{sattt}_{self.sensor}_{fn}/{row.overpass_datetime.year}/{row.overpass_datetime.month:02d}/{row.overpass_datetime.day:02d}/*'
            afiles += self.aws.glob(ponaws)
        # afiles
        
        afdf = _pd.DataFrame(afiles, columns=['p2faws'])
        # afdf
        
        #### get the two closest files that need to be downloaded
        
        # row = afdf.iloc[-1]
        
        def fn2times(row):
            times = row.p2faws.replace('.nc','').split('_')[-3:-1]
            times = [_pd.to_datetime(t[1:],format = '%Y%m%d%H%M%S%f') for t in times]
            return times
        
        # get scan start, end, and center from file name and set start as index.
        afdf[['scanstart', 'scanend']] = afdf.apply(fn2times, axis = 1, result_type='expand')
        afdf.index = afdf.scanstart
        afdf.sort_index(inplace=True)
        afdf['scanncenter'] = ((afdf.scanend - afdf.scanstart)/2) + afdf.scanstart

        self.tp_afdf = afdf
        
        #### if multiple version exist use only the newest!
        afdf['version'] = afdf.apply(lambda row: row.p2faws.split('/')[-1].split('_')[1], axis = 1)
        afdf.index.name = 'index'
        afdf = afdf.sort_values(['scanstart', 'version'])
        afdf = afdf[~ afdf.index.duplicated(keep = 'last')]
        
        # get the two closest to overpass
        closest = (afdf.scanncenter - srow.overpass_datetime).abs().sort_values().iloc[:2]
        closest.sort_index(inplace=True)
        # return closest
        closest_end = afdf.loc[closest.index[-1]].scanend
        closest_start = afdf.loc[closest.index[0]].scanstart
        clodest_interval = (closest_end - closest_start)/_pd.to_timedelta(1, 'minutes')
        download_df = afdf.loc[closest.index].p2faws.copy()
        if download_df.shape[0] > 2: #duplicates exist, probably due to different versions. the following will use only the newest version
            assert(False), 'this should no longer be an issue?!?'
            download_df = _pd.DataFrame(download_df)
            download_df['version'] = download_df.apply(lambda row: row.p2faws.split('/')[-1].split('_')[1], axis = 1)
            download_df = download_df.sort_values(['scanstart', 'version'])
            download_df = download_df[~ download_df.index.duplicated(keep = 'last')].p2faws
            clodest_interval = float(_np.unique(clodest_interval)) #remove multiple values
            closest_end = _pd.to_datetime(_np.unique(closest_end)[0])
            closest_start = _pd.to_datetime(_np.unique(closest_start)[0])
            
        self.tp_download_df = download_df
        self.tp_clodest_interval = clodest_interval
        self.tp_srow = srow
        self.tp_closest_end = closest_end
        self.tp_closest_start = closest_start
        if raise_error:
            assert(clodest_interval < 3), f'Interval to large. 2 files with NOAA 20 cover ~ 3 minutes. The ones that are the closest have a interval of {clodest_interval:0.1f} minutes. This probably means there is a data gap, like at night?'
            assert(clodest_interval > 2), f'Interval to small. 2 files with NOAA 20 cover ~ 3 minutes. The ones that are the closest have a interval of {clodest_interval:0.1f} minutes. This probably means the files are almoste empty, like at night?\n{download_df}'
            assert(closest_start < srow.overpass_datetime < closest_end), f'This happens if the file is right on the edge of the day. If it causes problems, fix it, Consider previous and next day.\not={srow.overpass_datetime}\nst={closest_start}\net={closest_end}'
        else:
            # print(f'closeest interval: {clodest_interval}')
            # print(type(clodest_interval))
            # print(download_df)
            # for idx,row in download_df.items():
            #     print(row.split('/')[-1])
    
            if not (clodest_interval < 3) or not (clodest_interval > 2) or not (closest_start < srow.overpass_datetime < closest_end):
                # download_aws = download_df.p2faws.copy()
                download_df[:] = _np.nan
                # return download_aws
        
        # details fo files to download
    
        # download_df['path2file_local'] = download_df.apply(lambda row: self.path2folder_local.joinpath(pl.Path(row.p2faws).name), axis = 1)
        # print(download_df.values)
        out = download_df.values
        if out.shape == (1,):
            out = _np.append(out, _np.nan)
        self.tp_out = out
        return out   
    
    
    
    @property
    def product(self):
        return self._product
    
    @product.setter
    def product(self, value):
        if value[-1] == self.scan_sector:
            warnings.warn('last letter of product is equal to scan sector ... by mistake?')
        self._product = value
        return
        
    def info_on_current_query(self):
        # if self.type_of_orbit == 'leo':
        #     p2faws = _np.unique(_np.concatenate([self.workplan.path2file_aws1.values, self.workplan.path2file_aws2.values]))
        #     nooffiles = p2faws.shape[0]
        # else:
        nooffiles = self.workplan.shape[0]
        if nooffiles == 0:
            info = 'no file found or all files already on disk.'
        else:
            du = self.estimate_disk_usage()
            disk_space_needed = du['disk_space_needed'] * 1e-6
            disk_space_free_after_download = du['disk_space_free_after_download']
            info = (f'no of files: {nooffiles}\n'
                    f'estimated disk usage: {disk_space_needed:0.0f} mb\n'
                    f'remaining disk space after download: {disk_space_free_after_download:0.0f} %\n')
        return info
    
    # def print_readme(self):
    #     url = 'https://docs.opendata.aws/noaa-goes16/cics-readme.html'
    #     html = _urllib.request.urlopen(url).read().decode("utf-8") 
    #     out = _html2text.html2text(html)
    #     print(out)
    
    def estimate_disk_usage(self, sample_size = 10): #mega bites
        step_size = int(self.workplan.shape[0]/sample_size)
        if step_size < 1:
            step_size = 1
        # if self.type_of_orbit == 'leo':
        #     #### TODO currently existing files are not considered here!! Needed diskspace might be overestimated
        #     p2faws = _np.unique(_np.concatenate([self.workplan.path2file_aws1.values, self.workplan.path2file_aws2.values]))
        #     sizes = _pd.DataFrame(p2faws, columns = ['fn'])[::10].apply(lambda row: self.aws.disk_usage(row.fn), axis = 1)
        #     disk_space_needed = sizes.mean() * p2faws.shape[0]
        sizes = self.workplan.iloc[::step_size].apply(lambda row: self.aws.disk_usage(row.path2file_aws), axis = 1)
        # sizes = self.workplan.iloc[::int(self.workplan.shape[0]/sample_size)].apply(lambda row: self.aws.disk_usage(row.path2file_aws), axis = 1)
        disk_space_needed = sizes.mean() * self.workplan.shape[0]
        
        # get remaining disk space after download
        du = _psutil.disk_usage(self.path2folder_local)
        disk_space_free_after_download = 100 - (100* (du.used + disk_space_needed)/du.total )
        out = {}
        out['disk_space_needed'] = disk_space_needed
        out['disk_space_free_after_download'] = disk_space_free_after_download
        return out
    
    @property
    def overpassplan(self):
        assert(self.type_of_orbit == 'leo'), 'Only used for polar orbiting satellites'
        if isinstance(self._overpassplan, type(None)):
            import atmPy.plattforms.satellites.orbits as atmorb
            
            #### get all scannes 
            ### (overpasses that hat the site in the field of view)
            
            days = _pd.date_range(self.start.date(), self.end.date(), freq='d')
            for es, site in enumerate(self.site):
                for e,d in enumerate(days):
                    dstr = f'{d.year:04d}{d.month:02d}{d.day:02d}'
                    if e ==0:
                        op = atmorb.OverPasses(satellite = self.satellite, start = dstr, site = site)
                    else:
                        op.start = dstr
                    opsel = op.overpasses.where(op.overpasses.observed_by_sensor).dropna('overpass_idx')
                    opsel = opsel.drop_vars('observed_by_sensor')
                    opsel = opsel.rename({'overpass_time_utc':'overpass_datetime'})
                    opsel = opsel.to_pandas()
                    if e == 0:
                        scanned = opsel
                    else:
                        scanned = _pd.concat([scanned, opsel])
                        
                scanned['site'] = site.abb
                try:
                    scanned[['path2file_aws1','path2file_aws2']] = scanned.apply(self._overpasstime2filesincloud, axis = 1, result_type='expand')
                except Exception as e:
                    print(scanned)
                    print(self._overpasstime2filesincloud(scanned.iloc[0]))
                    print(self._overpasstime2filesincloud(scanned.iloc[1]))
                    raise(e)
                    
                scanned.dropna(inplace=True)
                scanned.index = scanned.overpass_datetime
                scanned.drop('overpass_datetime', axis=1, inplace=True)
                if es == 0:
                    workplan = scanned
                else:
                    workplan = _pd.concat([workplan, scanned])
                    
            workplan.sort_index(inplace = True)
            workplan = workplan.truncate(self.start, self.end)
            if workplan.shape[0] > 0:
                workplan['path2file_local1'] = workplan.apply(lambda row: self.path2folder_local.joinpath(_pl.Path(row.path2file_aws1).name), axis = 1)
                workplan['path2file_local2'] = workplan.apply(lambda row: self.path2folder_local.joinpath(_pl.Path(row.path2file_aws2).name), axis = 1)
                workplan['path2file_aws1'] = workplan.apply(lambda row: _pl.Path(row.path2file_aws1), axis = 1)
                workplan['path2file_aws2'] = workplan.apply(lambda row: _pl.Path(row.path2file_aws2), axis = 1)
            else:
                workplan['path2file_local1'] = _np.nan
                workplan['path2file_local2'] = _np.nan
                
            self._overpassplan = workplan
        return self._overpassplan
    
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            if self._verbose:
                print('Get workplan:')
            if self.type_of_orbit == 'leo':
                df = _pd.DataFrame()
                df['path2file_aws'] = _pd.concat([self.overpassplan.path2file_aws1, self.overpassplan.path2file_aws2])#, ignore_index = True)
                df['path2file_local'] = _pd.concat([self.overpassplan.path2file_local1, self.overpassplan.path2file_local2])#, ignore_index = True)
                df.drop_duplicates(subset=['path2file_aws'], inplace=True)
                workplan = df
                workplan = workplan[~(workplan.apply(lambda row: row.path2file_local.is_file(), axis = 1))]
                # self._workplan = workplan
                # return self
            else:
                #### make a data frame to all the available files in the time range
                # create a dataframe with all hours in the time range
                df = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq='h'), columns=['path'])
                
                # create the path to the directory of each row above (one per houre)
                product_folder = self.path2folder_aws.joinpath(f'{self.product}{self.scan_sector}')
                df['path'] = df.apply(lambda row: product_folder.joinpath(str(row.name.year)).joinpath(f'{row.name.day_of_year:03d}').joinpath(f'{row.name.hour:02d}').joinpath('*'), axis= 1)
                # get the path to each file in all the folders 
                files_available = []
                
                #### TODO memory leak: below reload aws instance            
                # self.aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
                for idx,row in df.iterrows():
                    files_available += self.aws.glob(row.path.as_posix())
    
                #### Make workplan
    
                workplan = _pd.DataFrame([_pl.Path(f) for f in files_available], columns=['path2file_aws'])
                workplan['path2file_local'] = workplan.apply(lambda row: self.path2folder_local.joinpath(row.path2file_aws.name), axis = 1)
    
                #### remove if local file exists
                if not self._process:
                    if not self.overwrite:
                        workplan = workplan[~workplan.apply(lambda row: row.path2file_local.is_file(), axis = 1)]
                
                # get file sizes ... takes to long to do for each file
    #             workplan['file_size_mb'] = workplan.apply(lambda row: self.aws.disk_usage(row.path2file_aws)/1e6, axis = 1)
                
                #### get the timestamp
                def row2timestamp(row):
                    sos = row.path2file_aws.name.split('_')[-3]
                    assert(sos[0] == 's'), f'Something needs fixing, this string ({sos}) should start with s.'
                    ts = _pd.to_datetime(sos[1:-1],format = '%Y%j%H%M%S')
                    return ts
    
                workplan.index = workplan.apply(lambda row: row2timestamp(row), axis = 1)
    
            #### truncate ... remember so far we did not consider times in start and end, only the entire days
            workplan = workplan.sort_index()
            workplan = workplan.truncate(self.start, self.end)
            if workplan.shape[0] != 0:
                #### processing additions
                if self._process:
                    ### add path to processed file names
                    workplan["path2file_local_processed"] = workplan.apply(lambda row: self._process_path2processed.joinpath(f'{self._process_name_prefix}_{row.name.year}{row.name.month:02d}{row.name.day:02d}_{row.name.hour:02d}{row.name.minute:02d}{row.name.second:02d}.nc'), axis = 1)
                    ### remove if file exists 
                    workplan = workplan[~workplan.apply(lambda row: row.path2file_local_processed.is_file(), axis = True)]
                    # workplan['path2file_tmp'] = workplan.apply(lambda row: self._process_path2processed_tmp.joinpath(row.name.__str__()), axis = 1)
            
            else:
                if self._verbose:
                    print('workplan is empty')
                
            self._workplan = workplan
            if self._verbose:
                print('workplan done')
        return self._workplan       
    
    
    @workplan.setter
    def workplan(self, new_workplan):
        self._workplan = new_workplan
    
    @property
    def product_available_since(self):
        product_folder = self.path2folder_aws.joinpath(f'{self.product}{self.scan_sector}')
        years = self.aws.glob(product_folder.joinpath('*').as_posix())
        years.sort()
        
        is2000 = True
        while is2000:
            yearfolder = years.pop(0)
            firstyear = yearfolder.split('/')[-1]
            # print(firstyear)
            if firstyear != '2000':
                is2000 = False
                
        yearfolder = _pl.Path(yearfolder)
        days = self.aws.glob(yearfolder.joinpath('*').as_posix())
        days.sort()
        firstday = int(days[0].split('/')[-1])
        firstday_ts = _pd.to_datetime(firstyear) + _pd.to_timedelta(firstday, "D")
        return firstday_ts
        
    def download(self, test = False, overwrite = False, alternative_workplan = False,
                 error_if_low_disk_space = True):
        """
        

        Parameters
        ----------
        test : TYPE, optional
            DESCRIPTION. The default is False.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        alternative_workplan : pandas.Dataframe, optional
            This will ignore the instance workplan and use the provided one 
            instead. The default is False.
        error_if_low_disk_space : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        if isinstance(alternative_workplan, _pd.DataFrame):
            workplan = alternative_workplan
        else:
            workplan = self.workplan
        
        if error_if_low_disk_space:
            disk_space_free_after_download = self.estimate_disk_usage()['disk_space_free_after_download']
            assert(disk_space_free_after_download<90), f"This download will bring the disk usage above 90% ({disk_space_free_after_download:0.0f}%). Turn off this error by setting error_if_low_disk_space to False."
        
        for idx, row in workplan.iterrows():
            if not overwrite:
                if row.path2file_local.is_file():
                    continue
            #### TODO memory leak ... this did not help ... the following line is trying to deal with it. Its actaully not clear if the leak only happens when processing ... then the following line should not help
            # self.aws.clear_instance_cache() 
            # next try: reload aws instance: not helping
            # self.aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
            out = self.aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
            if test:
                break
        return out
    
    
    def process(self, raise_exception = False, verbose = False):
    # deprecated first grouping is required
        # group = self.workplan.groupby('path2file_local_processed')
        # for p2flp, p2flpgrp in group:
        #     break
        ## for each file in group
        if verbose:
            print(f'start processing ({self.workplan.shape[0]}): ', end = '')
        for dt, row in self.workplan.iterrows():
            if verbose:
                print('.', end = '')
            if row.path2file_local_processed.is_file():
                continue
            if not row.path2file_local.is_file():
    #             print('downloading')
                #### download
                # download_output = 
                
                #### TODO memory leak ... i did not notice that the download is done separately here... maybe try out the cach purch only
                # self.aws.clear_instance_cache()     #-> not helping           
                # self.aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True) - not helping
                self.aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
            #### process
            try:
                #### TODO memory leak check if row is the same before and after
                rowold = row.copy()
                self._process_function(row)
                if not row.equals(rowold):
                    print('row changed ... return')
                    return row, rowold
                if verbose:
                    print(':', end = '')
            except:
                if raise_exception:
                    raise
                else:
                    print(f'error applying function on one file {row.path2file_local.name}. The raw fill will still be removed (unless keep_files is True) to avoid storage issues')
            #### remove raw file
            if not self.keep_files:
                row.path2file_local.unlink()
            if verbose:
                print('|', end = '')
        if verbose:
            print('Done')      
        return
    
    #### TODO now since I am using multiprossing.Process instead of Pool we might want to separate the nesdis packages again.
    def process_parallel(self, process_function = None, args = {}, no_of_cpu = 2, 
                         raise_exception = False, 
                         path2log= None, 
                         subprocess = '',server = '', comment = '', 
                         verbose = True):
    # deprecated first grouping is required
        # group = self.workplan.groupby('path2file_local_processed')
        # for p2flp, p2flpgrp in group:
        #     break
        ## for each file in group
        if verbose:
            print(f'start processing ({self.workplan.shape[0]}): ', end = '')
        # for dt, row in self.workplan.iterrows():
        
  
        
  
        if 0:
            #### TODO this if can be removed
            # pool = mp.Pool(processes=no_of_cpu)
            pool = mp.get_context('spawn').Pool(processes=no_of_cpu)
            idx, rows = zip(*list(self.workplan.iterrows()))
            pool.map(partial(process_function, **args), rows)
            pool.close()
            pool.join()
        
        elif 0:
            #### TODO this elif can be removed
            # the following lead to a truncation of the workplan. Only when mod(len(workplan), no_of_cpu) = 0 all rows would be considered
            mp.set_start_method('spawn')
            if self.workplan.shape[0] == 0:
                if verbose:
                    print('workplan is empty, nothing to do here')
                return
            idx, rows = zip(*list(self.workplan.iterrows()))
            print('====idx')
            print(idx)
            print('====rows')
            print(rows)
            print(f'no_of_cpu: {no_of_cpu} {type(no_of_cpu)}')
            print(zip(*[list(l) for l in _np.array_split(rows, no_of_cpu)]))
                  
            for row_sub in zip(*[list(l) for l in _np.array_split(rows, no_of_cpu)]):
                print('=', end = '', flush = True)
                subproslist = []
                
                # print(f'row_sub: {row_sub}', flush = True)
                for row in row_sub:
                    rowt= rows[0].copy()
                    rowt.iloc[:] = row
                    row = rowt
                    print(',',  end = '', flush = True)
                    # print(f'row: {row}', flush = True)
                    process = mp.Process(target=process_function, args = (row,), kwargs = args)
                    process.start()
                    subproslist.append(process)
                [p.join() for p in subproslist]
                
                datetime = _pd.Timestamp.now()
                run_status = 1
                error = 0
                success = no_of_cpu
                warning = 0
                if not isinstance(path2log, type(None)):
                    with open(path2log, 'a') as log_out:
                            log_out.write(f'{datetime},{run_status},{error},{success},{warning},{subprocess},{server},{comment}\n')
                
                # break
            
        
        else:
            if isinstance(mp.get_start_method(allow_none=True), type(None)):
                mp.set_start_method('spawn')
            else:
                assert(mp.get_start_method() == 'spawn'), f'This should not be possible, we want to "spawn" new processes, not to "{mp.get_start_method()}" them.'

            self.workplan['grp'] = range(self.workplan.shape[0])
            self.workplan['grp'] +=1
            #no_of_cpu = 3
            self.workplan.grp /=no_of_cpu
            self.workplan.grp = _np.ceil(self.workplan.grp)
            
            run_status = 1
            warning = 0
            for idx, grp in self.workplan.groupby('grp'):
                print('=', end = '', flush = True)
                subproslist = []
                error_queue = mp.Queue()
                error = 0
                for grpidx, row in grp.iterrows():
                    print(',',  end = '', flush = True)
                    # print(f'row: {row}', flush = True)
                    process = mp.Process(target=process_function, args = (row,error_queue), kwargs = args)
                    process.start()
                    subproslist.append(process)
                [p.join() for p in subproslist]
                
                #### raise error if present
                while not error_queue.empty():
                    e = error_queue.get()
                    if isinstance(e, RuntimeError) and (str(e) == 'NetCDF: HDF error'):
                        print('RuntimeError, There was a problem reading the netcdf file. Clean-up required!!')
                        error += 1
                    else:
                        print("THIS SHOULD STOP EVERYTHING!!")
                        print(type(e))
                        print(str(e))
                        raise(e)   
                        
                datetime = _pd.Timestamp.now()
                
                success = no_of_cpu - error
                if not isinstance(path2log, type(None)):
                    with open(path2log, 'a') as log_out:
                            log_out.write(f'{datetime},{run_status},{error},{success},{warning},{subprocess},{server},{comment}\n')
                            
        if verbose:
            print('Done')
        return
        #### todo: concatenate 
        # if this is actually desired I would think this should be done seperately, not as part of this package
        # try:
        #     ds = _xr.open_mfdataset(p2flpgrp.path2file_tmp)

        #     #### save final product
        #     ds.to_netcdf(p2flp)
        
        #     #### remove all tmp files
        #     if not keep_tmp_files:
        #         for dt, row in p2flpgrp.iterrows():
        #             try:
        #                 row.path2file_tmp.unlink()
        #             except FileNotFoundError:
        #                 pass
        # except:
        #     print('something went wrong with the concatenation. The file will not be removed')

        



def test(f1):
    def f(x):
        f1(x)
        # f2(x)
    return f 

def process_row(row):#, process_function = None):
    # if verbose:
    #     print('.', end = '')
    return row #lambda x: x

    if row.path2file_local_processed.is_file():
        return
    if not row.path2file_local.is_file():
#             print('downloading')
        #### download
        # download_output = 
        
        #### TODO memory leak ... i did not notice that the download is done separately here... maybe try out the cach purch only
        # self.aws.clear_instance_cache()     #-> not helping           
        aws = _s3fs.S3FileSystem(anon=True, skip_instance_cache=True) #- not helping
        aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
        aws.close()
    #### process
    raise_exception = True
    try:
        #### TODO memory leak check if row is the same before and after
        rowold = row.copy()
        process_function(row)
        if not row.equals(rowold):
            print('row changed ... return')
            return row, rowold
        # if verbose:
        #     print(':', end = '')
    except:
        if raise_exception:
            raise
        else:
            print(f'error applying function on one file {row.path2file_local.name}. The raw fill will still be removed (unless keep_files is True) to avoid storage issues')
    #### remove raw file
    # if not self.keep_files:
    row.path2file_local.unlink()
    # if verbose:
    #     print('|', end = '')
