# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:36:07 2024

@author: busse
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']

#%% XGBoost, very strong model

# Define the model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'max_depth': [3, 5, 7],  # Maximum depth of a tree
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
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

# Best Parameters:  {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.6}
# Best MSE:  13.989654423576749
# Best R^2:  0.673507072853172
