#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:55:01 2025

@author: hagen
"""

import pandas as _pd
import xarray as _xr

from .. import file_io

class ChunckProcessor(object):
    def __init__(self, 
                 start = '20230101',
                 end = '20240101',
                 chunksize = 'MS',
                 sites = {'lon': -105.2705, 'lat': 40.015, 'alt': 1500, 'abb': 'bld', 'earthdata_granule': 'h09v04'}, 
                 p2fld_out = '~/tmp',
                 name_addon = 'projection',
                 scraper = None,
                 scraper_kwargs = None,
                 product = None,
                 product_kwargs = None,
                 overwrite = False,
                 remove_originals = True,
                 test = False,
                ):
        """
        This is a newer version of processor. The goal is to work by date chuncks (e.g. month). The workflow of this class is:
            1) Based on start, end and chuncksize a daterange is defined
            2) Pass a scraper (basically a interface with clouds et., e.g. glx.scrapers.earthdata.CMRSraperGlobal) in combination with its arguments
            3) Download all data fro a given chunk
            4) 

        Parameters
        ----------
        start : TYPE, optional
            DESCRIPTION. The default is '20230101'.
        end : TYPE, optional
            DESCRIPTION. The default is '20240101'.
        chunksize : TYPE, optional
            DESCRIPTION. The default is 'MS'.
        sites : TYPE, optional
            DESCRIPTION. The default is sites.
        p2fld_out : TYPE, optional
            DESCRIPTION. The default is p2fld_out.
        name_addon : TYPE, optional
            DESCRIPTION. The default is 'projection'.
        scraper : TYPE, optional
            DESCRIPTION. The default is None.
        scraper_kwargs : TYPE, optional
            DESCRIPTION. The default is None.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        test : TYPE, optional
            DESCRIPTION. The default is False.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.start = _pd.to_datetime(start)
        self.end = _pd.to_datetime(end)
        self.chunksize = chunksize
        self.scraper = scraper
        self.scraper_kwargs = scraper_kwargs
        self.product = product
        self.product_kwargs = product_kwargs
        self.name_addon = name_addon
        self.p2fld_out = p2fld_out
        self.sites = sites
        self.test = test
        self.remove_originals = remove_originals
        self.overwrite = overwrite
        self._masterplan = None

    @property
    def masterplan(self):
        if isinstance(self._masterplan, type(None)):
            mp = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq=self.chunksize,normalize = True), columns=['p2f_out'])
            
            def get_date(row):
                if self.chunksize in ['M', 'ME', 'MS']:
                    dt = f'{row.name.year:04d}{row.name.month:02d}'
                else:
                    assert(False), 'noep'
                return dt
            
            mp['p2f_out'] = mp.apply(lambda row: self.p2fld_out.joinpath(f'{self.scraper_kwargs['product']}_{self.name_addon}_{get_date(row)}.nc'), axis =1)
            end_time = mp.index[1:]
            mp = mp.iloc[:-1]
            mp['end_time'] = end_time - _pd.to_timedelta(1,'ns') #subtracting a nanosecond ensures that the interval is open on the right
            mp = mp.loc[:, ['end_time', 'p2f_out']]

            if self.test == 'full':
                mp = mp.iloc[[0],:]
            
            self._masterplan = mp
            
        return self._masterplan
    
    @masterplan.setter 
    def masterplan(self, value):
        self._masterplan = value

    @property
    def workplan(self):
        wp = self.masterplan
        if not self.overwrite:
            wp = wp[~wp.apply(lambda row: row.p2f_out.is_file(), axis = 1)]
        return wp
    
    @workplan.setter 
    def workplan(self, value):
        assert(False), 'workplan can not be set, alter the masterplan instead'

    def process_item(self, row, test = False, verbose = True):
        """
        

        Parameters
        ----------
        row : row
            A single row of the workplan. e.g. self.workplan.iloc[0]
        test : str, ['scraper']
            This will generate some test scenarios:
                scraper: This will return the initiated scraper. Allows you to 
                    see what will be downloaded for this particular step
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        scraper = self.scraper(start = row.name, end = row.end_time, **self.scraper_kwargs)
        if test == 'scraper':
            return scraper
        
        #### download all relevant files
        scraper.process()
        
        #### make the product (projection product that is)
        ds_list = []
        for sidx,srow in scraper.masterplan.iterrows():
            if verbose:
                print(f'opening: {srow.p2f_orig}')
            si = file_io.open_file(srow.p2f_orig, 
                                       # verbose = True
                                      )   
            ds = self.product(si, self.sites, 
                                              # test = True, 
                                              # verbose = True
                                           )
            # print(f'Is file still connected: {si.ds.encoding.get("source", None)}')
            si.ds.close()
            # print(f'Is file still connected: {si.ds.encoding.get("source", None)}')
            ds_list.append(ds)
        
        dsa = _xr.concat(ds_list, 'datetime')
        
        #### save chunk to netcdf
        if verbose:
            print(f'saving projection to {row.p2f_out}')
        dsa.to_netcdf(row.p2f_out)
        
        #### remove original files
        if self.remove_originals:
            for sidx,srow in scraper.masterplan.iterrows():
                srow.p2f_orig.unlink()
        return dsa
    
    def process(self, verbose = False):
        for idx, row in self.workplan.iterrows():
            self.process_item(row, verbose = verbose)