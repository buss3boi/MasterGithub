# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:22:13 2024

@author: busse
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

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
"""
# Use train_test_split to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% BASELINE MODEL


### Mean Method

# Calculate the average value of all the Z values. A model needs to be able to beat this
average_value = np.mean(y_train)

# Create a baseline vector with the same length as y_test filled with the average value
baseline_vector = np.full_like(y_test, fill_value=average_value)

# Calculate metrics for the baseline vector
baseline_mse = mean_squared_error(y_test, baseline_vector)
baseline_r2 = r2_score(y_test, baseline_vector)

# Print the results for the baseline model
print("Mean Baseline Model Metrics:")
print(f"Mean Squared Error (MSE): {baseline_mse} R^2 : {baseline_r2}")



### Median Method
median_value = np.median(y_train)

# Create a baseline vector with the same length as y_test filled with the average value
baseline_vector = np.full_like(y_test, fill_value=median_value)

# Calculate metrics for the baseline vector
baseline_mse = mean_squared_error(y_test, baseline_vector)
baseline_r2 = r2_score(y_test, baseline_vector)

# Print the results for the baseline model
print("Median Baseline Model Metrics:")
print(f"Mean Squared Error (MSE): {baseline_mse} R^2 : {baseline_r2}")




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

# Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2) score
mse = mean_squared_error(y_test, xtest_values)
r2 = r2_score(y_test, xtest_values)

# Print the results
print("Model performance")
print(f"Mean Squared Error (MSE): {mse} R^2 : {r2}")
"""

#%% K fold train_test_split cross validation

mse_dict = {'Mean':[], 'Median':[], 'Model':[]}
r2_dict = {'Mean':[], 'Median':[], 'Model':[]}
for i in range(42, 52):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    
    ### Mean Method
    
    # Calculate the average value of all the Z values. A model needs to be able to beat this
    average_value = np.mean(y_train)
    
    # Create a baseline vector with the same length as y_test filled with the average value
    baseline_vector = np.full_like(y_test, fill_value=average_value)
    
    # Calculate metrics for the baseline vector
    baseline_mse = mean_squared_error(y_test, baseline_vector)
    baseline_r2 = r2_score(y_test, baseline_vector)
    mse_dict['Mean'].append(baseline_mse)
    r2_dict['Mean'].append(baseline_r2)
    
    
    ### Median Method
    median_value = np.median(y_train)
    
    # Create a baseline vector with the same length as y_test filled with the average value
    baseline_vector = np.full_like(y_test, fill_value=median_value)
    
    # Calculate metrics for the baseline vector
    baseline_mse = mean_squared_error(y_test, baseline_vector)
    baseline_r2 = r2_score(y_test, baseline_vector)
    mse_dict['Median'].append(baseline_mse)
    r2_dict['Median'].append(baseline_r2)
    
    
    # Create an OrdinaryKriging instance
    OK = OrdinaryKriging(
        X_train[:, 0],  # X coordinates from training set
        X_train[:, 1],  # Y coordinates from training set
        y_train,        # Z values from training set
        variogram_model='hole-effect',  # Adjust variogram model as needed
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
    
    # Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2) score
    mse = mean_squared_error(y_test, xtest_values)
    r2 = r2_score(y_test, xtest_values)    
    mse_dict['Model'].append(mse)
    r2_dict['Model'].append(r2)

print("K fold CV Mean performance")
print(f"Mean Squared Error (MSE): {np.median(mse_dict['Mean'])} R^2 : {np.median(r2_dict['Mean'])}")
    
print("K fold CV Median performance")
print(f"Mean Squared Error (MSE): {np.median(mse_dict['Median'])} R^2 : {np.median(r2_dict['Median'])}")
    
print("K fold CV Model performance")
print(f"Mean Squared Error (MSE): {np.median(mse_dict['Model'])} R^2 : {np.median(r2_dict['Model'])}")
 

# All performances are a median of the 42 - 52 train test splits performance

# K fold CV Gaussian Model  performance
# Mean Squared Error (MSE): 21.27172452015609 R^2 : 0.4808400967324393

# K fold CV Exponential Model performance
# Mean Squared Error (MSE): 19.9634921344867 R^2 : 0.47563357313820387

# K fold CV Power Model performance
# Mean Squared Error (MSE): 20.935955199854625 R^2 : 0.4637147175475087

# K fold CV Spherical Model performance
# Mean Squared Error (MSE): 20.843682273424466 R^2 : 0.4631261882929422

# K fold CV Hole-Effect Model performance
# Mean Squared Error (MSE): 22.16746599186451 R^2 : 0.4322069664931372

# K fold CV Linear Model performance
# Mean Squared Error (MSE): 33.02282926712111 R^2 : 0.18453037795086807




# As we can see from the results there is not a significant difference in the 
# model performances hinting to the idea that it is not an important matter
