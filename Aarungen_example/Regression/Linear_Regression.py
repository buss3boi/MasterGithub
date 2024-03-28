# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:03:46 2024

@author: busse
"""

import pandas as pd
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from pykrige.ok import OrdinaryKriging

#%% Import File
# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'DQM')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)

OBS_XYZ_gdf = OBS_XYZ_gdf.rename(columns={"Shape_X": "X", "Shape_Y": "Y", "blengdber_": "Z"})


# Save file in current folder for simplicity
OBS_XYZ_gdf.to_csv('OBS_XYZ_gdf.csv', index=False)


#%% Model Linear Regressor (It sucks)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
from math import sqrt

# Load your dataset

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']

# Define the model
model = LinearRegression()

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    # Linear regression doesn't have hyperparameters to tune, so the grid is empty
}

# Define scoring methods (RMSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',
    'R^2': 'r2'
}

# Set up grid search
LR_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
LR_grid.fit(X, y)

# Print best parameters and best scores
print("Linear regresssion Best Parameters: ", LR_grid.best_params_)
print("Best MSE: ", -LR_grid.best_score_)
print("Best R^2: ", LR_grid.cv_results_['mean_test_R^2'][LR_grid.best_index_])



#%% Decision tree (Its alright, Similar to kriging)

from sklearn.tree import DecisionTreeRegressor


# Define the model
model = DecisionTreeRegressor(random_state=42)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'max_depth': [None, 10, 20, 30],  # Max depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Define scoring methods (RMSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',
    'R^2': 'r2'
}

# Set up grid search
DT_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
DT_grid.fit(X, y)

# Print best parameters and best scores
print("Decision tree Best Parameters: ", DT_grid.best_params_)
print("Best MSE: ", -DT_grid.best_score_)
print("Best R^2: ", DT_grid.cv_results_['mean_test_R^2'][DT_grid.best_index_])


# Decision tree Best Parameters:  {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2}
# Best MSE:  21.770629230031705
# Best R^2:  0.49626812770307877


#%% Support Vector Machine (This shit takes too long)

from sklearn.svm import SVR


# Define the model
model = SVR()

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'kernel': ['rbf'], # rbf is the default. Same results as sigmoid, poly 
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
}

# Define scoring methods (MSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',  # Note: We use neg_mean_squared_error to indicate higher scores are better
    'R^2': 'r2'
}

# Set up grid search
SVR_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
SVR_grid.fit(X, y)

# Print best parameters and best scores
print("SVM Best Parameters: ", SVR_grid.best_params_)
print("Best MSE: ", -SVR_grid.best_score_)  # Negate the score to get MSE
print("Best R^2: ", SVR_grid.cv_results_['mean_test_R^2'][SVR_grid.best_index_])


#%% KNN (Best model so far)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd



# Define the model
model = KNeighborsRegressor()

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_neighbors': [2, 3, 5, 7, 9],  # Number of neighbors to use
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'p': [1, 2]  # Power parameter for the Minkowski metric (1 for Manhattan distance, 2 for Euclidean distance)
}

# Define scoring methods (MSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',  # Note: We use neg_mean_squared_error to indicate higher scores are better
    'R^2': 'r2'
}

# Set up grid search
KNN_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
KNN_grid.fit(X, y)

# Print best parameters and best scores
print("KNN Best Parameters: ", KNN_grid.best_params_)
print("Best MSE: ", -KNN_grid.best_score_)  # Negate the score to get MSE
print("Best R^2: ", KNN_grid.cv_results_['mean_test_R^2'][KNN_grid.best_index_])


# KNN Best Parameters:  {'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
# Best MSE:  13.285184935092152
# Best R^2:  0.6883849064218859
