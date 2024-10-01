#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2024

@author: hagen

This script is to scrape data from the earth data system via the CMR API.
    
"""


#import nesdis_aws
# import nesdis_gml_synergy.satlab as ngs
import pathlib as pl
import warnings
#import socket
import argparse
import configparser
import pandas as pd
import ast
import numpy as np
import GeoLeoXtract as glx

import productomator.lab as prolab



warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")



def process_generic(product = 'MCD19A2v061',
                    stations = {'abb': 'TBL', 'name': 'Table Mountain (CO)', 'lat': 40.12498, 'lon': -105.2368},
                    satellite = 'TerraAqua',
                    sensor = 'MODIS',
                    # path2log = None, 
                    start='20200822 00:00:00',
                    end='20200828 00:00:00',
                    path2processed = '/nfs/stu3data2/Satellite_data/jpss/{satellite}/{product}/projections/surfrad',
                    file_prefix = 'projected2surfrad',
                    no_of_cpu = 1,
                    reporter = None,
                    verbose = False,
                    test = False
                    ):


    path2processed =  path2processed.format(product = product, satellite = satellite)
    pl.Path(path2processed).mkdir(exist_ok = True)

    pro = glx.scrapers.earthdata.CMRSraper(
                                            start=start,#'20120509 00:00:00', #20010201 00:00:00',#2000-02-24T00:00:00.000Z
                                            end=end,
                                            sites=stations,
                                            product=product,
                                            satellite=satellite,
                                            sensor=sensor,
                                            p2fld_out=path2processed, #'/home/grad/htelg/tmp/',
                                            prefix=file_prefix,
                                            reporter=reporter,
                                            overwrite=False,
                                            verbose=verbose,
                                        )

    pro.workplan = pro.workplan.sort_index(ascending = False)
    
    print(pro.workplan.iloc[0].p2f_out)
    # pro.workplan = pro.workplan.iloc[[1]]
    # pro.workplan = pro.workplan.iloc[:12]
    
    print('========', flush = True)
    print('workplan', flush = True)
    print('========', flush = True)
    print(f'workplan.shape: {pro.workplan.shape[0]}', flush = True)
    # if reverse:
    #     query.workplan = query.workplan[::-1]
    if test:
        print(pro.workplan, flush = True)
        return pro.workplan

    out = pro.process(max_processes=no_of_cpu, 
                       skip_no_granule_found_error=True,
                       skip_granule_missmatch_error=True,
                       skip_http_error = True,
                       skip_multiple_file_on_server_error = True)
    print('done', flush = True)
    return

#### if __name__ == '__main__':
def main():
    # #### create log file
    # fnlog = pl.Path('/home/grad/htelg/.processlogs/goes_aws_scraper_surfrad.log')
    # if not fnlog.is_file():
    #     with open(fnlog, 'w') as log_out:
    #         log_out.write('datetime,run_status,error,success,warning,subprocess,server,comment\n')
            
    #### argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--init', help = "Path to initiation file. All other arguments will be ignored.")
    # parser.add_argument('-n', '--todo', nargs='+', default=[], help = 'list of processes to run. To see option type jibarish here read the error message')
    # parser.add_argument('-c', '--cpus', help = 'number of cpus')
    # parser.add_argument('-s', '--start', help = 'start datetime, e.g. "2022-01-01 00:00:00"')
    # parser.add_argument('-e', '--end', help = 'end datetime, e.g. "2022-01-01 00:00:00"')
    # parser.add_argument('-v', '--verbose', help = 'verbose', action='store_true')
    # parser.add_argument('-r', '--reverse', help = 'reverse the workplan. Good if one wants to start a second process on the same product. Note, process tests each time if output file exists and will not try to re-process', action='store_true')
    # parser.add_argument('-t', '--test', action="store_true", help='will not execute at the end.')
    # parser.add_argument('-c', '--comment', help = 'some comment, e.g. which server this was stared on')
    args = parser.parse_args()

    
    
    #### TODO create log file
            
    #### initiation file parsing
    
    if not isinstance(args.init, type(None)):
        config = configparser.ConfigParser(allow_no_value=True,)
        config.read(args.init)
        if ('range' in config['workplan'].keys()) and ('start' in config['workplan'].keys()):
            assert(False), 'Initiation error:  give range OR start and end times.'
        
        if 'range' in config['workplan'].keys():
            tr = config['workplan']['range'].split()
            end = pd.Timestamp.now().date().__str__()
            start = (pd.Timestamp.now() - pd.to_timedelta(int(tr[1]), 'days')).date().__str__()
        elif 'start' in config['workplan'].keys():
            start = config['workplan']['start']
            end = config['workplan']['end']
        cpus = int(config['system']['cpus'])
        runtype = config['system']['runtype']
        products = [i.strip() for i in config['products']['products'].split(',')]
        test = 'testrun' in config['system'].keys()
        if test:
            test = config['system']['testrun'].split('#')[0].strip()
        verbose = 'verbose' in config['system'].keys()
        reverse = 'reverse' in config['workplan'].keys()
        path2processed =  config['file_io']['path2processed']
        file_prefix =  config['file_io']['file_prefix']
        reporter_name = f'cmrscraper_{args.init}'
        reporter_name = reporter_name.replace('.ini', '')
        sites = None
        network = None
        if 'network' in config['locations']:
            network = config['locations']['network']
            if network == 'surfrad':                
                import atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad.surfrad as surfrad
                sites = surfrad.network.stations.list
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
        satellites = [i.strip() for i in config['products']['satellite'].split('#')[0].split(',')]
        
        ### concat
        # concatenate = 'concatenate' in config['concatenate'].keys()
        # if concatenate:
        #     concat_rule = config['concatenate']['concatenate'].strip()
        #     concat_skip_last = 'skip_last' in  config['concatenate'].keys()
        
    else: 
        assert(False), 'args other than -i are currently not supported'
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
        print(f'start: {start}')
        print(f'end: {end}')
        print(f'cpus: {cpus}')
        print(f'products: {products}')
        print(f'test: {test}')
        print(f'reverse: {reverse}')
        print(f'runtype: {runtype}')
        print(f'path2processed: {path2processed}')
        print(f'file_prefix: {file_prefix}')
        print(f'network: {network}')
        print(f'sites: {sites}')
        print(f'satellites: {satellites}')
        print(f'reporter_name: {reporter_name}')
        # print(f'{}')
        # print(f'{}')
        print('==============')
    
           
    if runtype == 'generic':
        # test = True
        if test == 'True':
            print('just testing... generic')
           
        # if 1:
        #     pass
        else:        
            reporter = prolab.Reporter(reporter_name, 
                                       log_folder='/home/grad/htelg/.processlogs', 
                                       reporting_frequency=(1,'h'))
            for product in products:
                for satellite in satellites:
                    #### execute generic
                    process_generic(product = product,
                                    stations = sites,
                                    satellite = satellite,
                                    path2processed = path2processed,
                                    file_prefix = file_prefix,
                                    no_of_cpu = int(cpus),
                                    start = start,
                                    end = end,
                                    verbose = verbose,
                                    reporter = reporter,
                                    test = test == 'workplan')
    else:
        assert(False), f'runtype "{runtype}" is not a valid option.'


if __name__ == '__main__':
    main()