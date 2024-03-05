# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:22:13 2024

@author: busse
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
import pandas as pd


# import grid
from Granada_plotting import minx, maxx, miny, maxy, wxvec, wyvec


#%% Import obs_xyz_gdf



import os
import sys

# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'DQM')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)


# Assuming OBS_XYZ is a DataFrame with columns 'X', 'Y', 'Z'
# Replace this with the actual column names in your dataset
X = OBS_XYZ_gdf[['Shape_X', 'Shape_Y']].values
y = OBS_XYZ_gdf['blengdber_'].values

# Use train_test_split to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



#%% Ordinary kriging with train_test
# Create an OrdinaryKriging instance
OK = OrdinaryKriging(
    X_train[:, 0],  # X coordinates from training set
    X_train[:, 1],  # Y coordinates from training set
    y_train,        # Z values from training set
    variogram_model='exponential',  # Adjust variogram model as needed
    verbose=False
)


z_pred, ss = OK.execute('grid', wxvec, wyvec)



# Calculate scaling factors
scale_factor_x = (maxx - minx) / (z_pred.shape[1] - 1)
scale_factor_y = (maxy - miny) / (z_pred.shape[0] - 1)



# Calculate the indices in Wec for each point in Coords
indices_x = ((X_test[:, 0] - minx) / scale_factor_x).astype(int)
indices_y = ((X_test[:, 1] - miny) / scale_factor_y).astype(int)


xtest_values = []

# Iterate through each point in X_test and retrieve the corresponding value from z_pred
for i in range(len(X_test)):
    x_idx = indices_x[i]
    y_idx = indices_y[i]
    
    # Append the value from z_pred to the list
    xtest_values.append(z_pred[y_idx, x_idx])

# Convert the list to a NumPy array if needed
xtest_values = np.array(xtest_values)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2) score
mse, mae, r2 = mean_squared_error(y_test, xtest_values), mean_absolute_error(y_test, xtest_values), r2_score(y_test, xtest_values)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) score: {r2}")


