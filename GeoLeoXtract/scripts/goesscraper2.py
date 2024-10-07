#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:37:56 2022

@author: hagen

Done:
    DSR
    ACHA

from start to present ... running:
    COD (nimbus2 + tsunami2, 85662, 2/min/cpu -> 14 days 05/12)
    ACM (vortex4+nimbus3+nimbus4, 386050, 2/min/cpu -> 70 days 05/12)
    ADP (telg+pulsar4, 113568, 1/min/cpu -> 20 days 05/12)
    CTP (vortex2 + pulsar3    05/12)
    
"""


import GeoLeoXtract as glx
import pathlib as pl
import warnings
import socket
import argparse
import configparser
import pandas as pd
import ast
import numpy as np
import os
import productomator.lab as prolab


warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

# stations = surfrad.network.stations

# CPEX spring 2020
# start='2020-03-01 00:00:00',
# end='2020-06-01 00:00:00',
#
# CPEX original
# start='2018-08-01 00:00:00',
# end='2018-11-01 00:00:00',


#### generic
def process(product = 'ABI_L2_AOD',
                    stations = {'abb': 'TBL', 'name': 'Table Mountain (CO)', 'lat': 40.12498, 'lon': -105.2368},
                    satellite = 16,
                    # path2log = None, 
                    start='2017-04-20',
                    end='2023-01-01',
                    frequency = 'D',
                    # name = 'surfrad',
                    # path2processed =  '/nfs/stu3data2/Satellite_data/goes/{satellite}/ABI_L2_{product}_projected2surfrad/',
                    p2fld_out = '/nfs/stu3data2/Satellite_data/goes/{satellite}/{product}/projections/surfrad/v{version:03d}/concat',
                    p2fld_tmp = '/home/grad/htelg/tmp/',
                    # file_prefix = 'ABI_L2_{product}_projected2surfrad',
                    filenameformat = '{product}_projected2surfrad_{date}.nc',
                    no_of_cpu = 1,
                    version = 1,
                    reverse = False,
                    verbose = False,
                    test = False,
                    reporter = None,
                    ):
    if verbose:
        print('=========')
        print(f'satellite: {satellite}')
        print(f'product: {product}')
    # path2processed =  path2processed.format(product = product, satellite = satellite, version = version)
    p2fld_out = p2fld_out.format(satellite = satellite, product = product, 
                                 # name = name, 
                                 version = version)
    if 1:
        assert(pl.Path(p2fld_out).is_dir()), f"Output path does not exist ... generate it!:\n pl.Path({p2fld_out}).mkdir()"
        
    else:
        pt = pl.Path(p2fld_out)
        pt.parent.parent.mkdir(exist_ok = True)
        pt.parent.mkdir(exist_ok = True)
        pt.mkdir(exist_ok = True)
        # pl.Path(path2processed).mkdir(exist_ok = True)


    # product = ''
    # name = 'surfrad'
    # satellite = '16'
    # version = 1
    # p2fld_out = p.
    # pattern = '{product}_projected2{name}_{date}.nc' #ABI_L2_AOD_projected2surfrad_20230813.nc
    # p2fld_tmp = '/home/grad/htelg/tmp/'
    # frequency = 'D'
    # start = '2020-01-01'
    # end = '2024-01-01'
    # stations = atmsrf.network.stations
    
    scp = glx.scrapers.goes.GOESScraper(product=product,
                                          # name=name,
                                          satellite=satellite,
                                          p2fld_out=p2fld_out,
                                          pattern=filenameformat,
                                          p2fld_tmp=p2fld_tmp,
                                          frequency=frequency,
                                          start=start,
                                          end=end,
                                        stations = stations,
                                        reporter = reporter,
                                        )


    print('--------', flush = True)
    print('workplan', flush = True)
    print('--------', flush = True)
    print(f'workplan.shape: {scp.workplan.shape}  - {pd.Timestamp.now()}', flush = True)
    if reverse:
        scp.workplan = scp.workplan[::-1]
    if test:
        print(scp.workplan, flush = True)
        return scp.workplan
    
    scp.process(max_processes=no_of_cpu)

    print('done', flush = True)
    return

#### if __name__ == '__main__':
# if __name__ == '__main__':
def main():
    # #### create log file
    # fnlog = pl.Path('/home/grad/htelg/.processlogs/goes_aws_scraper_surfrad.log')
    # if not fnlog.is_file():
    #     with open(fnlog, 'w') as log_out:
    #         log_out.write('datetime,run_status,error,success,warning,subprocess,server,comment\n')
    
    #### settings - hardcoded
    version = 1 
    '''
    version 1
    ==========
    New:
        - products are tested for there version. The AOD in particular had a 
        change in version which came along a change in the qc-flags
    '''
       
    
    #### argparsing
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--init', help = "Path to initiation file. All other arguments will be ignored.")
    parser.add_argument('-n', '--todo', nargs='+', default=[], help = 'list of processes to run. To see option type jibarish here read the error message')
    parser.add_argument('-c', '--cpus', help = 'number of cpus')
    parser.add_argument('-s', '--start', help = 'start datetime, e.g. "2022-01-01 00:00:00"')
    parser.add_argument('-e', '--end', help = 'end datetime, e.g. "2022-01-01 00:00:00"')
    parser.add_argument('-v', '--verbose', help = 'verbose', action='store_true')
    parser.add_argument('-r', '--reverse', help = 'reverse the workplan. Good if one wants to start a second process on the same product. Note, process tests each time if output file exists and will not try to re-process', action='store_true')
    parser.add_argument('-t', '--test', action="store_true", help='will not execute at the end.')
    # parser.add_argument('-c', '--comment', help = 'some comment, e.g. which server this was stared on')
    args = parser.parse_args()
    print('args:')
    print(args)
    
    #### change process name
    # import prctl
    # prctl.set_name(f"{args.init.replace('.ini', '')}")
    

            
    #### initiation file parsing
    
    if not isinstance(args.init, type(None)):
        assert(pl.Path(args.init).is_file()), f'init file {args.init} does not exist'
        config = configparser.ConfigParser(allow_no_value=True,)
        config.read(args.init)
        if ('range' in config['workplan'].keys()) and ('start' in config['workplan'].keys()):
            assert(False), 'Initiation error:  give range OR start and end times.'
        
        if 'range' in config['workplan'].keys():
            tr = config['workplan']['range'].split()
            end = pd.Timestamp.now().date() - pd.to_timedelta(1, 'D')
            if tr[0] == 'all_time':
                start = '20170501'
            else:
                start = end- pd.to_timedelta(int(tr[1])-1, 'days')
        elif 'start' in config['workplan'].keys():
            start = config['workplan']['start']
            end = config['workplan']['end']
        cpus = int(config['system']['cpus'])
        # runtype = config['system']['runtype']
        products = [i.strip() for i in config['products']['products'].split(',')]
        test = 'testrun' in config['system'].keys()
        if test:
            test = config['system']['testrun'].split('#')[0].strip()
        verbose = 'verbose' in config['system'].keys()
        reverse = 'reverse' in config['workplan'].keys()
        path2processed =  config['file_io']['path2processed']
        path2tmp = config['file_io']['path2tmp']
        # file_prefix =  config['file_io']['file_prefix']
        filenameformat =  config['file_io']['filenameformat']
        reporter_name = f'goesscraper_{args.init}'
        reporter_name = reporter_name.replace('.ini', '')
        
        sites = None
        network = None
        if 'network' in config['locations']:
            network = config['locations']['network']
            if network == 'surfrad':                
                import atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad.surfrad as surfrad
                sites = surfrad.network.stations
        else: 
            requried_keys = ['abb', 'lon', 'lat']
            sites = []
            for k in config['locations'].keys():
                site = ast.literal_eval(f"{{{config['locations'][k]}}}")
                assert(isinstance(site, dict)), f'We expected site to be of type dict not "{type(site)}"'
                site['abb'] = k
                assert(np.all([k in site.keys() for k in requried_keys])), f'Site missing required info. Needed: {requried_keys}. Given: {site.keys()}'
                sites.append(site)
            if len(sites) == 0:
                sites = None
                
        assert(np.any([not isinstance(i, type(None)) for i in [sites,network]])), 'Init file does not contain sites or networks'
        satellites = [int(i) for i in config['products']['satellite'].split('#')[0].split(',')]
        
        
    else: 
        start = args.start
        end = args.end
        cpus = args.cpus
        products = args.todo
        test = args.test
        verbose = args.verbose
        reverse = args.reverse
        runtype = 'designated'
        path2processed =  '/nfs/stu3data2/Satellite_data/goes/16/ABI_L2_{product}_projected2surfrad/'
        file_prefix = 'ABI_L2_{product}_projected2surfrad',
        
    if verbose:
        print('========')
        print('Settings')
        print('========')
        print(f'pid: {os.getpid()}')
        print(f'start: {start}')
        print(f'end: {end}')
        print(f'cpus: {cpus}')
        print(f'products: {products}')
        print(f'test: {test}')
        print(f'reverse: {reverse}')
        # print(f'runtype: {runtype}')
        print(f'path2processed: {path2processed}')
        print(f'filenameformat: {filenameformat}')
        print(f'network: {network}')
        print(f'sites: {sites}')
        print(f'satellites: {satellites}')
        print(f'reporter_name: {reporter_name}')
        # print(f'{}')
        # print(f'{}')
        print('==============')
    
                     
    products = [p if "ABI" in p else f'ABI_L2_{p}' for p in products]
        # product = f'ABI_L2_{product}'
    assert(len(products) == 1), f'only a single productc can currently be past! given: {products}.'
    
    # test = True
    if test == 'True':
        print('just testing... ')
    # if 1:
    #     pass
    else:
        reporter = prolab.Reporter(reporter_name, 
                                   log_folder='/home/grad/htelg/.processlogs', 
                                   reporting_frequency=(1,'h'))
        for product in products:
            for satellite in satellites:
                #### execute generic
                process(product = product,
                        stations = sites,
                        satellite = satellite,
                        p2fld_out = path2processed,
                        p2fld_tmp = path2tmp,
                        filenameformat = filenameformat,
                        # path2log = fnlog,
                        no_of_cpu = int(cpus),
                        start = start,
                        end = end,
                        frequency = 'D',
                        verbose = verbose, 
                        reverse = reverse,
                        version = version,
                        reporter = reporter,
                        test = test == 'workplan')
    

