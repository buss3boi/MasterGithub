# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:51:51 2023

@author: busse

File on mapping the drill holes on a map. Code from Matlab will be used to create
this version on Python
"""
import geopandas as gpd
import pandas as pd
import numpy as np

# Define shapefile paths
fn_grunnvbronn = 'Grunnvannsborehull/GrunnvannBronn_20221130.shp'
fn_energibronn = 'Grunnvannsborehull/EnergiBronn_20221130.shp'
fn_bronnpark = 'Grunnvannsborehull/BronnPark_20221130.shp'
fn_oppkomme = 'Grunnvannsborehull/GrunnvannOppkomme_20221130.shp'
fn_sonderbor = 'Grunnvannsborehull/Sonderboring_20221130.shp'

# Read shapefiles using Geopandas
GRV = gpd.read_file(fn_grunnvbronn)
ENB = gpd.read_file(fn_energibronn)
BRP = gpd.read_file(fn_bronnpark)
OPK = gpd.read_file(fn_oppkomme)
SND = gpd.read_file(fn_sonderbor)

# Make a vertical correction for length to bedrock if borehole is not vertical
GRV['vdGRV'] = np.cos(np.radians(GRV['bhelnigrad'])) * GRV['blengdber_']
ENB['vdENB'] = np.cos(np.radians(ENB['bhelnigrad'])) * ENB['blengdber_']

# Horizontal 'migration' of x- and y- coordinates according to boretAzimut
GRV['hdGRV'] = np.sin(np.radians(GRV['bhelnigrad'])) * GRV['blengdber_']
ENB['hdENB'] = np.sin(np.radians(ENB['bhelnigrad'])) * ENB['blengdber_']

dxGRV = GRV['hdGRV'] * np.sin(np.radians(GRV['bazimuth']))
dyGRV = GRV['hdGRV'] * np.cos(np.radians(GRV['bazimuth']))
dxENB = ENB['hdENB'] * np.sin(np.radians(ENB['bazimuth']))
dyENB = ENB['hdENB'] * np.cos(np.radians(ENB['bazimuth']))

# Define the window coordinates
wminx = 254100
wmaxx = 268000
wminy = 6620100
wmaxy = 6628700
wxlength = wmaxx - wminx
wylength = wmaxy - wminy

# Create a meshgrid
dx = 100
dy = 100
wxvec = np.arange(wminx, wmaxx + dx, dx)
wyvec = np.arange(wminy, wmaxy + dy, dy)
wxgrid, wygrid = np.meshgrid(wxvec, wyvec)

nx, my = wxgrid.shape

grvx = GRV['Shape_X'] + dxGRV
grvy = GRV['Shape_Y'] + dyGRV

enbx = ENB['Shape_X'] + ENB['dxENB']
enby = ENB['Shape_Y'] + ENB['dyENB']


#%% Next section

# Filter out boreholes with blengdber_ == NaN or blengdber_ == 0
GRV = GRV[~(GRV['blengdber_'].isna() | (GRV['blengdber_'] == 0))]
ENB = ENB[~(ENB['blengdber_'].isna() | (ENB['blengdber_'] == 0))]

# Filter out wells outside the area of interest
OBS_XYZ = pd.concat([pd.DataFrame({'X': grvx, 'Y': grvy, 'Depth': GRV['blengdber_']}),
                    pd.DataFrame({'X': enbx, 'Y': enby, 'Depth': ENB['blengdber_']})])

OBS_XYZ = OBS_XYZ[(OBS_XYZ['X'] > wminx) & (OBS_XYZ['X'] < wmaxx) &
                  (OBS_XYZ['Y'] > wminy) & (OBS_XYZ['Y'] < wmaxy)]

