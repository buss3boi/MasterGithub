# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:52 2023

@author: busse
"""
import numpy as np
from pykrige.ok import OrdinaryKriging

# Define the observed data with float64 data type
data = np.array([
    [61.0, 139.0, 477.0],
    [63.0, 140.0, 696.0]
], dtype=np.float64)

x_obs = data[:, 0]
y_obs = data[:, 1]
z_obs = data[:, 2]

# Specify the target location with float64 data type
xo, yo = 65.0, 137.0

# Set up the Ordinary Kriging model
ok = OrdinaryKriging(
    x_obs,
    y_obs,
    z_obs,
    variogram_model='exponential',
    verbose=False
)

# Calculate the kriging estimate
z_est, ss_est = ok.execute("grid", [xo], [yo])

# The kriging estimate is stored in z_est[0], and the standard error in ss_est[0]

print(f"Kriging Estimate: {z_est[0]}")
print(f"Kriging Standard Error: {ss_est[0]}")
