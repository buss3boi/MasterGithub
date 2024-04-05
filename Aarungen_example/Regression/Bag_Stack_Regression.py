# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:23:19 2024

@author: busse
"""

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']


#%% Bagging
# Define the base estimator (in this case, Decision Tree Regressor)
base_estimator = DecisionTreeRegressor()

# Define the bagging regressor
bagging_model = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42)

# Evaluate using cross-validation
mse_scores = cross_val_score(bagging_model, X, y, scoring='neg_mean_squared_error', cv=5)
r2_scores = cross_val_score(bagging_model, X, y, scoring='r2', cv=5)

# Print the mean scores
print("Mean MSE:", -mse_scores.mean())
print("Mean R^2:", r2_scores.mean())

# Mean MSE: 26.190332668445883
# Mean R^2: 0.4002781352690513

#%% Stacking

from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

# Define the base estimators
base_estimators = [
    ('decision_tree', DecisionTreeRegressor()),
    ('linear_regression', LinearRegression())
]

# Define the stacking regressor
stacking_model = StackingRegressor(estimators=base_estimators, final_estimator=LinearRegression())

# Evaluate using cross-validation
mse_scores = cross_val_score(stacking_model, X, y, scoring='neg_mean_squared_error', cv=5)
r2_scores = cross_val_score(stacking_model, X, y, scoring='r2', cv=5)

# Print the mean scores
print("Mean MSE:", -mse_scores.mean())
print("Mean R^2:", r2_scores.mean())

# Mean MSE: 40.99103311753119
# Mean R^2: 0.07559364467309937

# Linear Regression + Decision Tree sucks. Try something else



#%% Ridge & Lasso regression (Do not include these trash methods)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd

# Define the model
model = Ridge()

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'alpha': [0.1, 1, 10],  # Regularization strength
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

# Best MSE:  40.91815118119697
# Best R^2:  0.05754795481064525


### Lasso 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd

model = Lasso()

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'alpha': [0.1, 1, 10],  # Regularization strength
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

# Best MSE:  40.91816082728237
# Best R^2:  0.057547661008972927