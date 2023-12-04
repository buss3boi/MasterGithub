# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:28:07 2023

@author: busse
"""

import numpy as np
import matplotlib.pyplot as plt
from Semivariogram_modeling import semivar_mod
from Semivario_meanOvari import funk_semivar_mean_var
import pandas as pd


# Load data from 'precipitation.txt' using pandas
df = pd.read_csv('precipitation.txt', sep='\t')

# Extract relevant columns
u1 = df['x-coord']
u2 = df['y-coord']
z = df['prec (mm/y)']


# Plot the scatter plot of x-coordinates against y-coordinates
plt.figure()
plt.plot(u1, u2, 'o')
plt.axis('equal')
plt.grid(True)
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.title('Precipitation Exercise')

# Define the input matrix Z1
Z1 = np.column_stack((u1, u2, z))

# Set parameters
maxdist = 17.8e3  # 17.8 km
ant = 8  # Number of classes in semivariogram

# Compute experimental semivariogram
hegam_precipitation = funk_semivar_mean_var(Z1, Z1, ant, maxdist)


# Extract lag distances (hlag)
hlag = hegam_precipitation[:, 1]

# Define parameters for the semivariogram model
C0 = 0.0
C1 = 6.0e4 - C0
a = 9000
pr = np.log(20)  # Practical range equal to 95% of total variance
eg = 1  # 1st order exponential

# Compute the theoretical semivariogram model
ch = semivar_mod(hlag, C0, C1, a, pr, eg)

# Plot experimental and model semivariograms
plt.figure()
plt.plot(hegam_precipitation[:, 1], hegam_precipitation[:, 2], 'ob', label='Experimental')
plt.plot(hlag, (C1 + C0) - ch, 'r', label='Model')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel('Lag Distance, h (m)')
plt.ylabel('γ(h)')
plt.title('Semivariogram for Precipitation Data')
plt.show()
