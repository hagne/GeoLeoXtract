import xarray as xr
import pathlib as pl
import numpy as np
# import cartopy.crs as ccrs
# import metpy 
# from scipy import interpolate
# from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
from pyproj import Proj
import urllib
from pyquery import PyQuery as pq
import pandas as pd
import matplotlib.pyplot as plt



class SatelliteDataQuery(object):
    def __init__(self):
        self._base_url = 'http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/goes16_download.cgi'
        self.path2savefld = pl.Path('/mnt/data/data/goes/')
        self.path2savefld.mkdir(exist_ok=True)
        html = urllib.request.urlopen(self._base_url).read()
        self._doc = pq(html)
        self.query = pd.DataFrame(columns=['source', 'satellite', 'domain', 'product', 'date', 'hour', 'minute', 'url2file', 'url_inter','inbetween_page'])
        self._url_inter = pd.DataFrame(columns=['url', 'sublinks'])
    @property
    def available_products(self):
        # available products
        products = [i for i in self._doc.find('select') if i.name == 'product'][0]
        products_values = [i.attrib["value"]for  i in products]
#         print('\n'.join([f'{i.attrib["value"]}:\t {i.text}' for  i in products]))
        return products_values
    @property
    def available_domains(self):
        domains = [i for i in self._doc.find('select') if i.name == 'domain'][0]
        domains_values = [i.attrib["value"]for  i in domains]
#         print('\n'.join([f'{i.attrib["value"]}:\t {i.text}' for  i in domains]))
        return domains_values

    @property
    def available_satellites(self):
        # available satellites
        satellites = [i for i in self._doc.find('input') if i.name == 'satellite']
        satellites_values = [i.attrib['value'] for i in satellites] #[i.attrib["value"][-2:] for i in satellites]
#         print('\n'.join([f'{i.attrib["value"][-2:]}: {i.attrib["value"]}' for  i in satellites]))
        return satellites_values

    def _attache_intermeidate_linkes(self):
        get_inter_url = lambda row: self._base_url + '?' + '&'.join([f'{i[0]}={i[1]}' for i in row.reindex(['source', 'satellite', 'domain', 'product', 'date', 'hour']).items()])
        self.query['url_inter'] = self.query.apply(get_inter_url, axis = 1)

    def _get_intermediate_pages(self):
        for idx,row in self.query.iterrows():
            intermediate_url =  row.url_inter
        #     break

            if intermediate_url not in self._url_inter.url.values:

                html_inter = urllib.request.urlopen(intermediate_url).read()
                doc_inter = pq(html_inter)

                sub_urls = []
                for link in doc_inter('a'):
                #     print(link.attrib['href'])
                    if 0:
                        if 'noaa-goes16.s3' in link.attrib['href']:
                            print(link.attrib['href'])
                            break
                    else:
                        if len(link.getchildren()) == 0:
                            continue
                        if not 'name' in link.getchildren()[0].attrib.keys():
                            continue

                        if link.getchildren()[0].attrib['name'] == 'fxx':
                            sub_urls.append(link.attrib['href'])
                #             print(link.attrib['href'])
                #             break

                sub_urls = pd.DataFrame(sub_urls, columns = ['urls'])

                sub_urls['datetime']= sub_urls.apply(lambda row: pd.to_datetime(row.urls.split('/')[-1].split('_')[-3][1:-3], format = '%Y%j%H%M'), axis=1)


                self._url_inter = self._url_inter.append(dict(url = intermediate_url,
                                           sublinks = sub_urls), ignore_index= True)
                assert(not isinstance(row.minute, type(None))), 'following minutes are available ... choose! '+ ', '.join([str(i.minute) for i in sub_urls.datetime])

            else:
                pass
#                 print('gibs schon')
                
    def _get_link2files(self):
        for idx,row in self.query.iterrows():
            sublinks = self._url_inter[self._url_inter.url == row.url_inter].sublinks.iloc[0]
            for sidx, srow in sublinks.iterrows():
                if srow.datetime.minute == int(row.minute):
                    row.url2file = srow.urls
                       
    def _generate_save_path(self):
        def gen_output_path(self,row):
            fld = self.path2savefld.joinpath(row['product']) 
            fld.mkdir(exist_ok=True)
            
            try: 
                p2f = fld.joinpath(row.url2file.split('/')[-1])
            except:
                print(f'promblem executing " p2f = fld.joinpath(row.url2file.split('/')[-1])" in {row}, with {fld}')
                assert(False)
            
            return p2f
        self.query['path2save'] = self.query.apply(lambda row: gen_output_path(self,row), axis = 1)
    
    @property
    def workplan(self):
        self._get_link2files()
        self._generate_save_path()
        self.query['file_exists'] = self.query.apply(lambda row: row.path2save.is_file(), axis = 1)
        self._workplan = self.query[~self.query.file_exists]
        return self._workplan
        
    def add_query(self, source = 'aws',
                     satellite = 'noaa-goes16',
                     domain = 'C',
                     product = 'ABI-L2-AOD',
                     date = '2020-06-27',
                     hour = 20,
                     minute = [21, 26]):
        
        if not isinstance(minute, list):
            minute = [minute]
        if not isinstance(hour, list):
            hour = [hour]
            
        for qhour in hour:
            for qmin in minute:
                qdict = dict(source = source,
                         satellite = satellite,
                         domain = domain,
                         product = product,
                         date = date,
                         hour = f'{qhour:02d}',
                         minute = f'{qmin:02d}'
                         )
                self.query = self.query.append(qdict, ignore_index = True)
        assert(satellite in self.available_satellites)
        assert(domain in self.available_domains)
        assert(product in self.available_products)
        
        self._attache_intermeidate_linkes()
        self._get_intermediate_pages()
        # drop line if minute is None
        for idx, row in self.query.iterrows():
            if isinstance(row.minute, type(None)):
                self.query.drop(idx, inplace=True)
                
    def download_query(self, test = False):
        for idx, row in self.workplan.iterrows():
            print(f'downloading {row.url2file}', end = ' ... ')
            if row.path2save.is_file():
                print('file already exists ... skip')
            else:
                urllib.request.urlretrieve(row.url2file, filename=row.path2save)
                print('done')
            if test:
                break


class GeosSatteliteProducts(object):
    def __init__(self,file):
        if type(file) == xr.core.dataset.Dataset:
            ds = file
        else:
            ds = xr.open_dataset(file)
        
        self.ds = ds
        
#         self._varname4test = 'CMI_C02'
        self._lonlat = None
        
    @property
    def lonlat(self):
        if isinstance(self._lonlat, type(None)):
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
            p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

            # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
            # to latitude and longitude values.
            XX, YY = np.meshgrid(x, y)
            lons, lats = p(XX, YY, inverse=True)
            
            # Assign the pixels showing space as a single point in the Gulf of Alaska
#             where = np.isnan(self.ds[self._varname4test].values)
            where = np.isinf(lons)
            lats[where] = 57
            lons[where] = -152

            self._lonlat = (lons, lats) #dict(lons = lons, 
#                                     lats = lats)
        return self._lonlat
            # Assign the pixels showing space as a single point in the Gulf of Alaska
    #             where = np.isnan(channels_rgb['red'])
    #             lats[where] = 57
    #             lons[where] = -152

class or_abi_l2_mcmipc(GeosSatteliteProducts):
    def __init__(self, *args):
        super().__init__(*args)
#         self._varname4test = 'CMI_C02'

    
    def plot_true_color(self, 
                        gamma = 1.8,#2.2, 
                        contrast = 130, #105
                        projection = None,
                        bmap = None
                       ):
        channels_rgb = dict(red = self.ds['CMI_C02'].data.copy(),
                            green = self.ds['CMI_C03'].data.copy(),
                            blue = self.ds['CMI_C01'].data.copy())
        
        channels_rgb['green_true'] = 0.45 * channels_rgb['red'] + 0.1 * channels_rgb['green'] + 0.45 * channels_rgb['blue']

        
        
        for chan in channels_rgb:
            col = channels_rgb[chan]
            # Apply range limits for each channel. RGB values must be between 0 and 1
            new_col = col / col[~np.isnan(col)].max()
            
            # apply gamma
            if not isinstance(gamma, type(None)):
                new_col = new_col**(1/gamma)
            
            # contrast
            #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
            if not isinstance(contrast, type(None)):
                cfact = (259*(contrast + 255))/(255.*259-contrast)
                new_col = cfact*(new_col-.5)+.5
            
            channels_rgb[chan] = new_col
            
        rgb_image = np.dstack([channels_rgb['red'],
                             channels_rgb['green_true'],
                             channels_rgb['blue']])
        rgb_image = np.clip(rgb_image,0,1)
            
        a = plt.subplot()
        if isinstance(projection, type(None)) and isinstance(bmap, type(None)):
            
            a.imshow(rgb_image)
#             a.set_title('GOES-16 RGB True Color', fontweight='semibold', loc='left', fontsize=12);
#             a.set_title('%s' % scan_start.strftime('%d %B %Y %H:%M UTC '), loc='right');
            a.axis('off')
    
        else:          
            lons,lats = self.lonlat
            
            # Make a new map object Lambert Conformal projection
            if not isinstance(bmap,Basemap):
                bmap = Basemap(resolution='i', projection='aea', area_thresh=5000, 
                             width=3000*3000, height=2500*3000, 
    #                          lat_1=38.5, lat_2=38.5, 
                             lat_0=38.5, lon_0=-97.5)

                bmap.drawcoastlines()
                bmap.drawcountries()
                bmap.drawstates()

            # Create a color tuple for pcolormesh

            # Don't use the last column of the RGB array or else the image will be scrambled!
            # This is the strange nature of pcolormesh.
            rgb_image = rgb_image[:,:-1,:]

            # Flatten the array, becuase that's what pcolormesh wants.
            colortuple = rgb_image.reshape((rgb_image.shape[0] * rgb_image.shape[1]), 3)

            # Adding an alpha channel will plot faster, according to Stack Overflow. Not sure why.
            colortuple = np.insert(colortuple, 3, 1.0, axis=1)

            # We need an array the shape of the data, so use R. The color of each pixel will be set by color=colorTuple.
            pc = bmap.pcolormesh(lons, lats, channels_rgb['red'], color=colortuple, linewidth=0, latlon=True, zorder = 0)
            pc.set_array(None) # Without this line the RGB colorTuple is ignored and only R is plotted.

#             plt.title('GOES-16 True Color', loc='left', fontweight='semibold', fontsize=15)
#             plt.title('%s' % scan_start.strftime('%d %B %Y %H:%M UTC'), loc='right');
