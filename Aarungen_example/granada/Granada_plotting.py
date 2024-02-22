""" 
File for loading granada dataset. The dataset will be saved as OBS_XYZ. 
This file contains X, Y and Z values for the Granada data. Vertical and
Horizontal corrections has been made. Use grid wyvecx, wyvecy,
edit the grid size with dx, dy
"""
import os
print(os.getcwd())

import geopandas as gpd
import pandas as pd
import numpy as np

#%% Read Granada shapefiles, read geotables
fn_grunnvbronn = 'Grunnvannsborehull/GrunnvannBronn_20221130.shp'
fn_energibronn = 'Grunnvannsborehull/EnergiBronn_20221130.shp'

GRV = gpd.read_file(fn_grunnvbronn)
ENB = gpd.read_file(fn_energibronn)

# Section 2: Define the bounding box
wminx, wmaxx = 254100, 268000
wminy, wmaxy = 6620100, 6628700

# Set the grid box size
dx, dy = 100, 100

wxvec = np.arange(wminx, wmaxx + dx, dx, dtype=np.float64)
wyvec = np.arange(wminy, wmaxy + dy, dy, dtype=np.float64)

wxgrid, wygrid = np.meshgrid(wxvec, wyvec)

# Section 3: Data Preprocessing

# Vertical migration of non vertical points
vdGRV = np.cos(np.radians(GRV['bhelnigrad'].astype(float))) * GRV['blengdber_']
vdENB = np.cos(np.radians(ENB['bhelnigrad'].astype(float))) * ENB['blengdber_']

# Horizontal migration
hdGRV = np.sin(np.radians(GRV['bhelnigrad'].astype(float))) * GRV['blengdber_']
hdENB = np.sin(np.radians(ENB['bhelnigrad'].astype(float))) * ENB['blengdber_']

dxGRV = hdGRV * np.sin(np.radians(GRV['bazimuth'].astype(float)))
dyGRV = hdGRV * np.cos(np.radians(GRV['bazimuth'].astype(float)))

dxENB = hdENB * np.sin(np.radians(ENB['bazimuth'].astype(float)))
dyENB = hdENB * np.cos(np.radians(ENB['bazimuth'].astype(float)))

# Update the coordinates

# Update GRV and ENB with vertical corrections
for idx, row in GRV.iterrows():
    if not np.isnan(vdGRV[idx]):
        GRV.at[idx, 'blengdber_'] = vdGRV[idx]

for idx, row in ENB.iterrows():
    if not np.isnan(vdENB[idx]):
        ENB.at[idx, 'blengdber_'] = vdENB[idx]

def GRVupdate_shape(row):
    if dxGRV[row.name] > 0:
        return row.geometry.x + dxGRV[row.name], row.geometry.y + dyGRV[row.name]
    else:
        return row.geometry.x, row.geometry.y

GRV[['Shape_X', 'Shape_Y']] = GRV.apply(GRVupdate_shape, axis=1, result_type="expand")


def ENBupdate_shape(row):
    if dxENB[row.name] > 0:
        return row.geometry.x + dxENB[row.name], row.geometry.y + dyENB[row.name]
    else:
        return row.geometry.x, row.geometry.y

ENB[['Shape_X', 'Shape_Y']] = ENB.apply(ENBupdate_shape, axis=1, result_type="expand")


# Filter out boreholes with blengdber_ == NaN or blengdber_ == 0
OBS_GRV = GRV[(~GRV['blengdber_'].isna()) & (GRV['blengdber_'] > 0)][['Shape_X', 'Shape_Y', 'blengdber_']]
OBS_ENB = ENB[(~ENB['blengdber_'].isna()) & (ENB['blengdber_'] > 0)][['Shape_X', 'Shape_Y', 'blengdber_']]

# Combine the dataframes
OBS_XYZ = pd.concat([OBS_GRV, OBS_ENB], axis=0)

# Filter out wells outside the area of interest
OBS_XYZ = OBS_XYZ[(OBS_XYZ['Shape_X'] >= wminx) & (OBS_XYZ['Shape_X'] <= wmaxx) &
                  (OBS_XYZ['Shape_Y'] >= wminy) & (OBS_XYZ['Shape_Y'] <= wmaxy)]

# print(OBS_XYZ)
# Print test. PS: it works

OBS_XYZ.to_csv('OBS_XYZ.csv', index=False)