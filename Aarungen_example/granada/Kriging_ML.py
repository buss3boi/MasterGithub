# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:22:13 2024

@author: busse
"""

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from pykrige.ok import OrdinaryKriging
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

# import grid
from Granada_plotting import minx, maxx, miny, maxy, wxvec, wyvec
#from Kriging_manual import fwxvec, fwyvec, final_data


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


#%% Random_States Train_test_split validation baseline

mse_dict = {'Mean':[], 'Median':[], 'Model':[]}
r2_dict = {'Mean':[], 'Median':[], 'Model':[]}
random_states = range(42, 52)

def calc_random_state(random_states):
    for i in random_states:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        
        ### Mean Method
        
        # Calculate the average value of all the Z values. A model needs to be able to beat this
        average_value = np.mean(y_test) # y or y_test. Depends on the method
        # The baseline methods scores equally shit wether it is y_test or straight y
        
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
        
        
#%% K-Fold Cross validation
# Variogram with custom parameters 'range': 5661   'sill': 50   'nugget': 0
variogram_parameters = {'range': 2573, 'sill': 50, 'nugget': 1}
  
def validate_kriging(model, variogram_parameters, X, y, wxvec, wyvec):
    mse_dict['Model'] = []
    r2_dict['Model'] = []
    
    minx, maxx, = X[:, 0].min(), X[:, 0].max()
    miny, maxy, = X[:, 1].min(), X[:, 1].max()
        
    ss = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
            
        # Create an OrdinaryKriging instance
        OK = OrdinaryKriging(
            X_train[:, 0],  # X coordinates from training set
            X_train[:, 1],  # Y coordinates from training set
            y_train,        # Z values from training set
            variogram_model=model,  # Adjust variogram model as needed
            verbose=False,
            # variogram_parameters = variogram_parameters, # COMMENT OUT, if no params wanted, Note! With universal params, probably scores better
            anisotropy_angle=-45,
            anisotropy_scaling=2.2,
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
        
    print(f"K fold CV {OK.variogram_model} Model performance")
    print(f"Mean Squared Error (MSE): {np.median(mse_dict['Model'])} R^2 : {np.median(r2_dict['Model'])}")

simple_model = validate_kriging('exponential', variogram_parameters, X, y, wxvec, wyvec)  


#iso_model = validate_kriging('exponential', variogram_parameters, final_data, y, fwxvec, fwyvec)   

# Performances of No parameters, 5 K-Fold cross validation, random state 42
# All so far in minx, maxx, miny, maxy grid

# As we can see from the results there is not a significant difference in the 
# model performances hinting to the idea that it is not an important matter


# K fold CV Gaussian Model  performance
# Mean Squared Error (MSE): 18.66744320276828 R^2 : 0.5776519306859222

# K fold CV Exponential Model performance
# Mean Squared Error (MSE): 19.413058900893333 R^2 : 0.49039571659593295

# K fold CV Spherical Model performance
# Mean Squared Error (MSE): 21.004360612666225 R^2 : 0.4495966743722781



### The following models had pre trained semivariogram on the kriging 
# estimates, allowing them to take information from all samples. This in turn
# gives them an advantage


# Exponential variogram_parameters = {'range': 3600, 'sill': 47, 'nugget': 6}
# Mean Squared Error (MSE): 15.868838570425993 R^2 : 0.6579035100750341

# Exponential variogram_parameters = {'range': 4450, 'sill': 50, 'nugget': 8}
# Mean Squared Error (MSE): 15.870299156675134 R^2 : 0.6637436800480508

# Exponential variogram auto = {'range': 2275, 'sill': 43, 'nugget': 9}
# Mean Squared Error (MSE): 15.79534507375323 R^2 : 0.6555443932512088

# Gaussian variogram_parameters = {'range': 2853.6, 'sill': 33, 'nugget': 17}
# Mean Squared Error (MSE): 23.084210885922303 R^2 : 0.41581616496977114

# Gaussian variogram_parameters = {'range': 3869, 'sill': 35, 'nugget': 20}
# Mean Squared Error (MSE): 29.035078953682696 R^2 : 0.31322368230399633

# Gaussian variogram auto = {'range': 1841, 'sill': 42.7, 'nugget': 3}
# Mean Squared Error (MSE): 18.747748904050944 R^2 : 0.5789548894759752 # Does not score better than pretrained. Interesting




### Results no set max values. Dataset decides bounds. A more fair way to decide since our
# machine learning algorithms uses this method


# Exponential Model no parameters
# Mean Squared Error (MSE): 15.105036307839448 R^2 : 0.6082944361952713

# Gaussian Model no parameters 
# Mean Squared Error (MSE): 19.741580605701273 R^2 : 0.59946555250808



# Exponential auto = {'range': 2275, 'sill': 43, 'nugget': 9}
# Mean Squared Error (MSE): 14.446088453440906 R^2 : 0.6845731379708244

# Exponential  curve_fit auto = {'range': 1826.09, 'sill':49, 'nugget': 3}
# Mean Squared Error (MSE): 14.440706870059156 R^2 : 0.680266329914254

# Exponential variogram_parameters = {'range': 4450, 'sill': 50, 'nugget': 8}
# Mean Squared Error (MSE): 15.264512394915675 R^2 : 0.6783273520739201


# Gaussian auto = {'range': 1841, 'sill': 42.7, 'nugget': 3}
# Mean Squared Error (MSE): 19.282778547339188 R^2 : 0.5935331608773511




### Pykrige manual Rotation (Exponential)

# Pykrige manual Rotation no paramters
# Mean Squared Error (MSE): 19.069245273672514 R^2 : 0.5257915472285244

# Pykrige manual rotation curve_fit = {'range': 1403.38, 'sill': 45.2, 'nugget': 1}
# Mean Squared Error (MSE): 14.00396335934763 R^2 : 0.6339027939245951

# Pykrige manual rotation auto = {'range': 5661, 'sill': 50, 'nugget': 1}
# Mean Squared Error (MSE): 14.109764050732473 R^2 : 0.6244632682171374



### Scaled data, scaled grid (Exponential)

# Scaled data + grid rotation no parameters
# Mean Squared Error (MSE): 22.537610612376998 R^2 : 0.49189835829971684

# Scaled data + grid rotation curve_fit = {'range': 1403.38, 'sill': 45.2, 'nugget': 1}
# Mean Squared Error (MSE): 13.337889181805641 R^2 : 0.6508671516088331

# Scaled data + grid rotation auto = {'range': 2573, 'sill': 50, 'nugget': 1}
# Mean Squared Error (MSE): 13.109747763082838 R^2 : 0.6510790817093608