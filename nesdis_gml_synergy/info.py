#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:04:05 2023

@author: hagen
"""

ABI_product_info = {'ABI-L1b-Rad': 'Radiances',
                 'ABI-L2-ACHA': 'Cloud Top Height',
                 'ABI-L2-ACHA2KM': 'Cloud Top Height 2km res. (todo: verify!!)',
                 'ABI-L2-ACHP2KM': 'Cloud Top Pressure 2km res. (todo: verify)',
                 'ABI-L2-ACHT': 'Cloud Top Temperature',
                 'ABI-L2-ACM': 'Clear Sky Mask',
                 'ABI-L2-ACTP': 'Cloud Top Phase',
                 'ABI-L2-ADP': 'Aerosol Detection',
                 'ABI-L2-AICE': 'Ice Concentration and Extent',
                 'ABI-L2-AITA': 'Ice Age and Thickness',
                 'ABI-L2-AOD': 'Aerosol Optical Depth',
                 'ABI-L2-BRF': 'Land Surface Bidirectional Reflectance Factor () 2 km resolution & DQFs',
                 # 'ABI-L2-CCL': 'unknown',
                 'ABI-L2-CMIP': 'Cloud and Moisture Imagery',
                 'ABI-L2-COD': 'Cloud Optical Depth',
                 'ABI-L2-COD2KM': 'Cloud Optical Depth 2km res.',
                 'ABI-L2-CPS': 'Cloud Particle Size',
                 'ABI-L2-CTP': 'Cloud Top Pressure',
                 'ABI-L2-DMW': 'Derived Motion Winds',
                 'ABI-L2-DMWV': 'L2+ Derived Motion Winds',
                 'ABI-L2-DSI': 'Derived Stability Indices',
                 'ABI-L2-DSR': 'Downward Shortwave Radiation',
                 'ABI-L2-FDC': 'Fire (Hot Spot Characterization)',
                 'ABI-L2-LSA': 'Land Surface Albedo () 2km resolution & DQFs',
                 'ABI-L2-LST': 'Land Surface Temperature',
                 'ABI-L2-LST2KM': 'Land Surface Temperature',
                 'ABI-L2-LVMP': 'Legacy Vertical Moisture Profile',
                 'ABI-L2-LVTP': 'Legacy Vertical Temperature Profile',
                 'ABI-L2-MCMIP': 'Cloud and Moisture Imagery',
                 'ABI-L2-RRQPE': 'Rainfall Rate (Quantitative Precipitation Estimate)',
                 'ABI-L2-RSR': 'Reflected Shortwave Radiation Top-Of-Atmosphere',
                 'ABI-L2-SST': 'Sea Surface (Skin) Temperature',
                 'ABI-L2-TPW': 'Total Precipitable Water',
                 'ABI-L2-VAA': 'Volcanic Ash: Detection and Height'}

VIIRS_product_info = {'AEROSOL_AOD_EN': 'AOD',
                      }

satellite_list = [dict(names = ['NOAA 20', 'NOAA_20'], 
                       type_of_orbit = 'leo'),
                  dict(names = ['GOES 16', 'G16',], 
                       type_of_orbit = 'goes'),
                  dict(names = ['GOES 17', 'G17',], 
                       type_of_orbit = 'goes'),
                  dict(names = ['GOES 18', 'G18',], 
                       type_of_orbit = 'goes'),
                  ]