# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:17:32 2024

@author: busse
"""


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']

#%% Neural network method. This method sucks
# Define the model
model = MLPRegressor(random_state=42)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {
    'hidden_layer_sizes': [(100, 50), (50, 50), (100, 50, 100)],  # Size of hidden layers
    'activation': ['relu', 'tanh', 'logistic'],  # Activation function
    'solver': ['adam', 'sgd'],  # Optimizer
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty parameter
    'learning_rate': ['constant', 'adaptive']  # Learning rate schedule
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


#%% Test with parameters with hidden layer

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model

hidden_layers = (225, 125, 75, 25)

model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='tanh', solver='adam', alpha=0.001, learning_rate='constant', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("R-squared (R^2): ", r2)


# Neural Networks sucks on this dataset