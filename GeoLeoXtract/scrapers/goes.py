import pathlib as _pl 
import pandas as _pd
import xarray as _xr

import multiprocessing
import psutil
import time

import warnings as _warnings
import GeoLeoXtract as _glx

class GOESScraper(object):
    def __init__(self,product = 'ABI_L2_AOD',
                 # name = 'surfrad',
                 satellite = '16',
                 p2fld_out = '/home/grad/htelg/tmp',
                 pattern = '{product}_projected2surfrad_{date}.nc', #ABI_L2_AOD_projected2surfrad_20230813.nc
                 p2fld_tmp = '/home/grad/htelg/tmp/',
                 frequency = 'D',
                 start = '2020-01-01',
                 end = '2024-01-01',
                 stations = None,
                 reporter = None,
                ):
        self.p2fld_out = _pl.Path(p2fld_out)
        self.p2fld_tmp = _pl.Path(p2fld_tmp)
        self.pattern = pattern
        self.product = product
        # self.name = name
        self.satellite = satellite
        self.start = start
        self.end = end
        self.frequency = frequency
        self.reporter = reporter
        self.stations = stations
        self._wp = None
        
    @property
    def workplan(self):
        if isinstance(self._wp, type(None)):
            #### make Workplan
            wp = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq = self.frequency), columns=['p2f_out'])
            wp.index.name = 'datetime'

            def row2path(row):
                date = f'{row.name.year:04d}{row.name.month:02d}{row.name.day:02d}'
                fn = self.pattern.format(product=self.product, 
                                         # name = self.name, 
                                         date = date)
                path = self.p2fld_out.joinpath(fn)
                return path

            # create file_path for output
            wp['p2f_out'] = wp.apply(lambda row: row2path(row), axis = 1)

            # remove files from wp when they exist
            wp = wp[~(wp.apply(lambda row: row.p2f_out.is_file(), axis = 1))]
            
            self._wp = wp

        return self._wp

    @workplan.setter
    def workplan(self,value):
        self._wp = value
        return

    def process_single_chunk(self,row, 
                             error_queue = None,
                             verbose = False, 
                             surpress_warnings = True,
                            ):
        
        if surpress_warnings:        
            _warnings.filterwarnings('ignore')
            
        cstart = row.name
        cend = cstart + _pd.to_timedelta(1,self.frequency)
        
        try:
            query = _glx.cloud_interface.AwsQuery(path2folder_local='/home/grad/htelg/tmp',
                                  satellite= self.satellite, #16,
                                  product=self.product.replace('_', '-'),
                                  scan_sector='C',
                                  start=cstart,
                                  end=cend,
                                  # process=None,
                                  overwrite=True,
                                )            
            query.download()
            
            # generate the path with the projected files
            # query.workplan['path2file_local_projected'] = query.workplan.apply(lambda qrow: qrow.path2file_local.parent.joinpath(qrow.path2file_local.name.replace('.nc', '_projected.nc')), axis = 1)
                        
            # process all downloaded files
            projections = []
            for idx, qrow in query.workplan.iterrows():
                print('.', end = '', flush = True)
                ds = _glx.products.projections.project_statellite2stations_v01(path2file_in=qrow.path2file_local, 
                                                 stations = self.stations,
                                                 # path2file_out=qrow.path2file_local_projected,
                                                )
                projections.append(ds)
            
            # comcatinate the projections
            dsc = _xr.concat(projections, 'datetime')
            
            # save concatinated file
            dsc.to_netcdf(row.p2f_out)
            
            # clean-up, delete 
            for idx, qrow in query.workplan.iterrows():
                qrow.path2file_local.unlink()

        except Exception as e:
            if verbose:
                print(e)
            if isinstance(error_queue, type(None)):
                raise 
            error_queue.put(e)
        return 
    
    def process(self, max_processes = 2,
                timeout = 60*60*6, sleeptime = 1,
                verbose = True,
                test = True,
                # skip_multiple_file_on_server_error = False,
               ):          
        
        iterator = self.workplan.iterrows()
        process_this = self.process_single_chunk           
        
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
                # if isinstance(e, GranuleMissmatchError):
                #     if skip_granule_missmatch_error:
                #         do_raise = False
                #         msg = 'GME'
                # elif isinstance(e, NoGranuleFoundError):
                #     if skip_granule_missmatch_error:
                #         do_raise = False
                #         msg = 'NGFE'
                        
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
                    idx,arg = next(iterator)
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
                                                  # kwargs={'skip_granule_missmatch_error': skip_granule_missmatch_error,
                                                  #         'skip_no_granule_found_error': skip_no_granule_found_error,
                                                  #         'skip_http_error': skip_http_error,
                                                  #         'skip_multiple_file_on_server_error': skip_multiple_file_on_server_error,
                                                  #        },  # keyword arguments 
                                                  name = 'jpssscraper')
                process.daemon = True
                processes.append(process)
                process.start()
                print('.', end = '')
                
        #### final report
        if not isinstance(self.reporter, type(None)):
            self.reporter.log(overwrite_reporting_frequency=True)        
        