"""
This is a collection of functions that help unify data products
"""

import pandas as _pd
import GeoLeoXtract as _glx

def project_statellite2stations_v01(path2file_in, stations, path2file_out = None, test = False, verbose = False):
    """
    Version 1. Projects satellite data to specific coordinates and their surrounding.

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
    ngsinst = _glx.satlab.open_file(path2file_in)
    
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
    ds.attrs['info'] = ('This file contains a projection of satellite data onto specific sites.\n'
                         'It includes the closest pixel data as well as the average over circular\n'
                         'areas with various radii. Note, for the averaged data only data is\n'
                         'considered with a qulity flag given by the prooduct class in the\n'
                         'GeoLeoXtract library.')
    

    # save2file
    if test:
        return ds
    if not isinstance(path2file_out, type(None)):
        ds.to_netcdf(path2file_out)
        # Memory kept on piling up -> maybe a cleanup will help
        ds.close()
    ngsinst.ds.close()
    ngsinst = None
    
    return ds