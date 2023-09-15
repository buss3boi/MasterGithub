# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:02:37 2023

@author: busse
"""

#%% Plotting 

# Link to plotting: https://betterprogramming.pub/creating-topographic-maps-in-python-convergence-of-art-and-data-science-7492b8c9fa6e

import rasterio
import pandas as pd
file = rasterio.open('6602_2_10m_z33.dem')
dataset = file.read()

print(dataset.shape)

from rasterio.plot import show

show(file, cmap='terrain')

Z = file.read(1)

df_z = pd.DataFrame(Z) ## FIGURE THIS OUT!!!!!

geotransform = file.transform

"""
# Coordinates for the 'window' you want to 'slice out' from the dataset
wminx = 254100
wmaxx = 268000
wminy = 6620100
wmaxy = 6628700

# Grid spacing
dx = 10
dy = 10


import numpy as np
import matplotlib.pyplot as plt

wxvec = np.arange(wminx, wmaxx + dx, dx)
wyvec = np.arange(wminy, wmaxy + dy, dy)

wxgrid, wygrid = np.meshgrid(wxvec, wyvec)


xv = np.arange(geotransform[0], geotransform[0] + geotransform[1] * Z.shape[1], dx)
yv = np.arange(geotransform[3], geotransform[3] + geotransform[5] * Z.shape[0], dy)
xcoord, ycoord = np.meshgrid(xv, yv)

minx, maxx, miny, maxy = np.min(xcoord), np.max(xcoord), np.min(ycoord), np.max(ycoord)

# Find indices of the crop area
i, j = np.where((xcoord >= wminx) & (xcoord <= wmaxx) & (ycoord >= wminy) & (ycoord <= wmaxy))

xc_ut = xcoord[i, j]
yc_ut = ycoord[i, j]
Zut = Z[i, j]

wi, wj = wxgrid.shape
wZ = Zut.reshape(wi, wj)

# Plot the original and cropped images
plt.figure()
plt.imshow(Z, extent=[minx, maxx, miny, maxy], cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar()
plt.plot(wminx, wminy, 'xr')
plt.plot(wmaxx, wmaxy, 'xr')
plt.axis('equal')

plt.figure()
plt.imshow(wZ, extent=[wminx, wmaxx, wminy, wmaxy], cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar()
plt.plot(wminx, wminy, 'xr')
plt.plot(wmaxx, wmaxy, 'xr')
plt.axis('equal')

plt.show()
"""