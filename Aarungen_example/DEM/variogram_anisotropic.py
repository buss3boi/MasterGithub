# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:40:54 2024

@author: Bastian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from pprint import pprint
plt.style.use('ggplot')

def load_dem(file_path):
    src = rasterio.open(file_path)
    dataset = src.read()
    
    geotransform = src.transform


    return dataset, src, geotransform


def plot_dem(dem_array):
    plt.imshow(dem_array, cmap='viridis')
    plt.colorbar(label='Elevation')
    plt.title('Digital Elevation Model (DEM)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


#%% Load and setup

dem_file_path = '6602_2_10m_z33.dem'

dem_array, src, geotransform = load_dem(dem_file_path)
dem_array= np.squeeze(dem_array)

# Specify the min and max values for X and Y
minx, maxx, miny, maxy = 254100, 268000, 6620100, 6628700

window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, geotransform)
   
# Read the data from the window
z_matrix = src.read(1, window=window)

plot_dem(z_matrix)


#%% Create the Variogram from GSTAT
import skgstat as skg
from skgstat import models
from scipy.optimize import curve_fit



num_samples = 5000
sampled_indices = np.random.choice(range(z_matrix.size), size=num_samples, replace=False)
sampled_coordinates = np.column_stack(np.unravel_index(sampled_indices, z_matrix.shape))

# Extract z values corresponding to sampled points
sampled_values = z_matrix[sampled_coordinates[:, 0], sampled_coordinates[:, 1]]

"""
# Make the variogram
variogram_model = skg.Variogram(sampled_coordinates, sampled_values, model="spherical")

variogram_model.plot()

# This looks like shit dont plot it XD 
# variogram_model.distance_difference_plot()


#%% Variogram models


# set estimator back
variogram_model.estimator = 'matheron'
variogram_model.model = 'spherical'

xdata = variogram_model.bins
ydata = variogram_model.experimental



# initial guess - otherwise lm will not find a range
p0 = [np.mean(xdata), np.mean(ydata), 0]
cof, cov =curve_fit(models.spherical, xdata, ydata, p0=p0)

print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))

### Spherical
xi =np.linspace(xdata[0], xdata[-1], 100)
yi = [models.spherical(h, *cof) for h in xi]

plt.plot(xdata, ydata, 'og')
plt.plot(xi, yi, '-b');
plt.title('Manual Semivariogram')



variogram_model.fit_method ='trf'


pprint(variogram_model.parameters)

variogram_model.fit_method ='lm'

variogram_model.plot();
pprint(variogram_model.parameters)


#%% Exponential model
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

axes[0].set_title('Spherical')
axes[1].set_title('Exponential')

variogram_model.fit_method = 'trf'
variogram_model.plot(axes=axes[0], hist=False)

# switch the model
variogram_model.model = 'exponential'
variogram_model.plot(axes=axes[1], hist=False);


# Try gaussian
variogram_model.model = 'gaussian'
variogram_model.plot();


# Matern
variogram_model.model = 'matern'
variogram_model.plot();


#%% Metrics

variogram_model.model = 'spherical'
rmse_sph = variogram_model.rmse
r_sph = variogram_model.describe().get('effective_range')

# exponential
variogram_model.model = 'exponential'
rmse_exp = variogram_model.rmse
r_exp = variogram_model.describe().get('effective_range')


variogram_model.model = 'gaussian'
rmse_gau = variogram_model.rmse
r_gau = variogram_model.describe().get('effective_range')


variogram_model.model = 'matern'
rmse_mat = variogram_model.rmse
r_mat = variogram_model.describe().get('effective_range')


print('Spherical   RMSE: %.2f' % rmse_sph)
print('Exponential RMSE: %.2f' % rmse_exp)
print('Gauss       RMSE: %.2f' % rmse_gau)
print('Matern      RMSE: %.2f' % rmse_mat)



print('Spherical effective range:    %.1f' % r_sph)
print('Exponential effective range:  %.1f' % r_exp)
print('Gauss effective range:        %.1f' % r_gau)
print('Matern effective range:       %.1f' % r_mat)
"""

#%% Directional Variogram

Vnorth = skg.DirectionalVariogram(sampled_coordinates, sampled_values, azimuth=90, tolerance=60, maxlag=0.5, n_lags=20)

Veast = skg.DirectionalVariogram(sampled_coordinates, sampled_values, azimuth=0, tolerance=60, maxlag=0.5, n_lags=20)

Vnowe = skg.DirectionalVariogram(sampled_coordinates, sampled_values, azimuth=45, tolerance=60, maxlag=0.5, n_lags=20)

Vnoea = skg.DirectionalVariogram(sampled_coordinates, sampled_values, azimuth=135, tolerance=60, maxlag=0.5, n_lags=20)

df_direction = pd.DataFrame({'north':Vnorth.describe(), 'east': Veast.describe()})


"""
fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
ax.plot(Vnoea.bins, Vnoea.experimental, '.--g', label='Noea-Sowe')
ax.plot(Vnowe.bins, Vnowe.experimental, '.--y', label='Nowe-Soea')

ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')
"""

#%% Create semi-variograms

def fit_and_plot_variogram(variogram, model_func, initial_guess=None):
    xdata = variogram.bins
    ydata = variogram.experimental

    # Initial guess - if not provided, use default values
    if initial_guess is None:
        initial_guess = [np.mean(xdata), np.mean(ydata), 0]

    # Fit the model to the variogram data
    coef, cov = curve_fit(model_func, xdata, ydata, p0=initial_guess)

    # Print model parameters
    print('Model: {}    range: {:.2f}   sill: {:.1f}   nugget: {:.2f}'.format(model_func.__name__, coef[0], coef[1], coef[2]))

    # Generate y values for the fitted model
    xi = np.linspace(xdata[0], xdata[-1], 100)
    yi = [model_func(h, *coef) for h in xi]

    # Plot the variogram data and the fitted model
    # plt.plot(xdata, ydata, 'og', label='Experimental')
    # plt.plot(xi, yi, '-b', label='Fitted Model')

    # # Set plot title indicating directional variogram and model
    # plt.title(f'{variogram}')
    # plt.legend()
    # plt.show()
    
    return xi, yi, coef
    

# Make the plots 

North_semivar = fit_and_plot_variogram(Vnorth, models.spherical)
North_semivar_exp = fit_and_plot_variogram(Vnorth, models.exponential)
North_semivar_gauss = fit_and_plot_variogram(Vnorth, models.gaussian)

East_semivar = fit_and_plot_variogram(Veast, models.spherical)
East_semivar_exp = fit_and_plot_variogram(Veast, models.exponential)
East_semivar_gauss = fit_and_plot_variogram(Veast, models.gaussian)

Noea_semivar = fit_and_plot_variogram(Vnoea, models.spherical)
Noea_semivar_exp = fit_and_plot_variogram(Vnoea, models.exponential)
Noea_semivar_gauss = fit_and_plot_variogram(Vnoea, models.gaussian)

Nowe_semivar = fit_and_plot_variogram(Vnowe, models.spherical)
Nowe_semivar_exp = fit_and_plot_variogram(Vnowe, models.exponential)
Nowe_semivar_gauss = fit_and_plot_variogram(Vnowe, models.gaussian)




#%% Plot back semivariograms
"""
# North south
fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
ax.plot(North_semivar[0], North_semivar[1], '-y', label='North semivar spherical')
ax.plot(North_semivar_exp[0], North_semivar_exp[1], '-b', label='North semivar exp')
ax.plot(North_semivar_gauss[0], North_semivar_gauss[1], '-g', label='North semivar gauss')

ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')


# East west
fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(Veast.bins, Veast.experimental, '.--r', label='East - West')
ax.plot(East_semivar[0], East_semivar[1], '-y', label='East semivar spherical')
ax.plot(East_semivar_exp[0], East_semivar_exp[1], '-b', label='East semivar exp')
ax.plot(East_semivar_gauss[0], East_semivar_gauss[1], '-g', label='East semivar gauss')

ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')



# From what we can see. The spherical semivariogram hits the best in most cases
# overall. We are still going to use a grid search to find optimal params
"""


fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
ax.plot(Vnoea.bins, Vnoea.experimental, '.--g', label='Noea-Sowe')
ax.plot(Vnowe.bins, Vnowe.experimental, '.--y', label='Nowe-Soea')

ax.plot(North_semivar[0], North_semivar[1], '-r', label='No-So semivar')
ax.plot(East_semivar[0], East_semivar[1], '-b', label='Ea-We semivar')
ax.plot(Noea_semivar[0], Noea_semivar[1], '-g', label='Noea-sowe semivar')
ax.plot(Nowe_semivar[0], Nowe_semivar[1], '-y', label='Nowe-soea semivar')


ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')
plt.savefig('Directional_semivariogram.png', dpi=300, bbox_inches='tight')


#Vnorth.pair_field(plt.gca())





