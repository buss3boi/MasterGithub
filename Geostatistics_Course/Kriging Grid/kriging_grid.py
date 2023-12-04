# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:29:24 2023

@author: busse
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Data
data = np.array([
    [1, 7000, 4000, 620.00],
    [2, 3000, 4000, 590.00],
    [3, -2000, 5000, 410.00],
    [4, -10000, 1000, 390.00],
    [5, -3000, -3000, 1050.00],
    [6, -7000, -7000, 980.00],
    [7, 2000, -3000, 600.00],
    [8, 2000, -10000, 410.00],
    [9, 0, 0, 810.00]
])

stations = data[:, 0]
x_obs = data[:, 1]
y_obs = data[:, 2]
z_obs = data[:, 3]

# Grid generation
grid_spacing = 100
x_min = x_obs.min() - 500
x_max = x_obs.max() + 500
y_min = y_obs.min() - 500
y_max = y_obs.max() + 500

x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, grid_spacing), np.arange(y_min, y_max, grid_spacing))

# Interpolation using Nearest Neighbor
tree = cKDTree(np.column_stack((x_obs, y_obs)))
dists, indices = tree.query(np.column_stack((x_grid.ravel(), y_grid.ravel())), k=1)
z_nearest_neighbor = z_obs[indices]

# Interpolation using Scipy's griddata with 'nearest' method
points = np.column_stack((x_obs, y_obs))
values = z_obs
interp_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
z_griddata_nearest = griddata(points, values, interp_points, method='nearest')

# Interpolation using Scipy's griddata with 'linear' method
z_griddata_linear = griddata(points, values, interp_points, method='linear')

# Plot results
def plot_grid(x, y, z, title):
    plt.figure()
    plt.pcolormesh(x, y, z, shading='auto')
    plt.colorbar()
    plt.plot(x_obs, y_obs, 'ro')
    plt.title(title)

plot_grid(x_grid, y_grid, z_nearest_neighbor.reshape(x_grid.shape), 'Nearest Neighbor Interpolation')
plot_grid(x_grid, y_grid, z_griddata_nearest.reshape(x_grid.shape), 'Scipy GridData (Nearest)')
plot_grid(x_grid, y_grid, z_griddata_linear.reshape(x_grid.shape), 'Scipy GridData (Linear)')

plt.show()
