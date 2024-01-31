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

wminx, wmaxx, wminy, wmaxy = 254100, 268000, 6620100, 6628700

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



