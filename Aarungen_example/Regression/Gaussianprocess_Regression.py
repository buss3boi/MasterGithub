# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:52:26 2024

@author: busse
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']


#%% Gaussian Process regressor
# Define the model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
model = GaussianProcessRegressor(kernel=kernel)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    "kernel": [C(1.0, (1e-3, 1e3)) * RBF(length_scale) for length_scale in [1, 10, 100]],  # Kernel parameters
    "alpha": [1e-10, 1e-5, 1e-2, 0.1, 1.0]  # Regularization parameter
}

# Define scoring methods (MSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',  # Note: We use neg_mean_squared_error to indicate higher scores are better
    'R^2': 'r2'
}

# Set up grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
grid_search.fit(X, y)

# Print best parameters and best scores
print("Best Parameters: ", grid_search.best_params_)
print("Best MSE: ", -grid_search.best_score_)  # Negate the score to get MSE
print("Best R^2: ", grid_search.cv_results_['mean_test_R^2'][grid_search.best_index_])

# Best Parameters:  {'alpha': 1.0, 'kernel': 1**2 * RBF(length_scale=10)}
# Best MSE:  32.80129055957692
# Best R^2:  0.24242829184851886