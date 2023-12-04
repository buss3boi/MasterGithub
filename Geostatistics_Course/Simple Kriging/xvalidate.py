# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:29:21 2023

@author: busse
"""
import numpy as np
from pykrige.ok import OrdinaryKriging

# Parameters
C0 = 0  # nugget
C1 = 500 - C0  # sill
a = 2000  # range
pr = np.log(1 - 0.95)  # practical range at 95% of C1 + C0

# Data
X = np.array([
    [1, 7000, 4000, 620.00],
    [2, 3000, 4000, 590.00],
    [3, -2000, 5000, 410.00],
    [4, -10000, 1000, 390.00],
    [5, -3000, -3000, 1050.00],
    [6, -7000, -7000, 980.00],
    [7, 2000, -3000, 600.00],
    [8, 2000, -10000, 410.00],  # Commented out
])

Xv = np.array([9, 0, 0, 810.00])
st, xo, yo, zo = Xv

stations = X[:, 0]  # Station IDs
x_obs = X[:, 1]  # X-coordinates of gauge stations
y_obs = X[:, 2]  # Y-coordinates of gauge stations
z_obs = X[:, 3]  # Precipitation values (mm/year)

# Create an Ordinary Kriging object
ok = OrdinaryKriging(x_obs, 
                     y_obs, 
                     z_obs, 
                     variogram_model="exponential",
                      variogram_parameters={"sill": C1, "range": a, "nugget": C0})

# Perform the kriging estimation
z_est, ss_est = ok.execute("grid", [xo], [yo])

# Calculate kriging variance
kriging_variance = ss_est[0]

# Print the results
print(f"Station ID: {st}")
print(f"X-coordinate: {xo}")
print(f"Y-coordinate: {yo}")
print(f"Observed Precipitation: {zo}")
print(f"Kriging Estimate: {z_est[0]}")
print(f"Kriging Variance: {kriging_variance}")
