# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:02:37 2023

@author: busse
"""

#%% Plotting 

# Link to plotting: https://betterprogramming.pub/creating-topographic-maps-in-python-convergence-of-art-and-data-science-7492b8c9fa6e

import rasterio
import pandas as pd
from rasterio.plot import show
import matplotlib.pyplot as plt

src = rasterio.open('6602_2_10m_z33.dem')
dataset = src.read()

print(dataset.shape)

from rasterio.plot import show
show(src, cmap='terrain')

geotransform = src.transform

wminx = 254100
wmaxx = 268000
wminy = 6620100
wmaxy = 6628700

window = rasterio.windows.from_bounds(wminx, wminy, wmaxx, wmaxy, geotransform)
   
# Read the data from the window
subset = src.read(1, window=window)
show(subset, cmap='terrain')


window_coords = [(wminx, wminy), (wminx, wmaxy), (wmaxx, wminy), (wmaxx, wmaxy)]

# Create a single figure and axis for both the image and the red dots
fig, ax = plt.subplots()

# Show the original DEM data with cmap 'terrain' on the same axis
show(src, cmap='terrain', ax=ax)

# Show the subset of the DEM data with cmap 'terrain' on the same axis
show(subset, cmap='terrain', ax=ax)

# Plot red dots at the corner coordinates on the same axis
for x, y in window_coords:
    row, col = src.index(x, y)
    ax.plot(col, row, 'ro', markersize=5)  # 'ro' denotes red circle markers

# Show the combined image with red dots
plt.show()
#With this code, both the original image and the red dots will be displayed on the same plot, and the red dots will appear on top of the image.








"""



# Grid spacing
dx = 10
dy = 10
"""