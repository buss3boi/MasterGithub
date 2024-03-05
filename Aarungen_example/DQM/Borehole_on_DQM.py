# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:19:18 2024

@author: busse
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Specify the path to your shapefile (without the file extension)
shapefile_path = "./Losmasse/LosmasseFlate_20221201.shp"

# Load the shapefile into a GeoDataFrame
gdf = gpd.read_file(shapefile_path)


gdf.plot(column='jordart', figsize=(18,11), legend=True)
plt.title('DQM file')
plt.savefig('DQM_file.png', dpi=300, bbox_inches='tight')
#jestem gupi 
# Now, the variable 'gdf' contains your shapefile data

minx, maxx = 254100, 268000
miny, maxy = 6620100, 6628700

# Filter the GeoDataFrame based on the bounding box
gdf_filtered = gdf.cx[minx:maxx, miny:maxy]
gdf_filtered.plot(column='jordart', figsize=(20,12), legend=False)
plt.title('Filtered DQM file')

# Define the classes you want to display

# Definition 1:
bedr_cover = [12, 13, 43, 100, 101, 110, 130, 140]
# Not all of these classes exists within the dataset

# Filter the GeoDataFrame to include only the desired classes
gdf_filtered_classes = gdf_filtered[gdf_filtered['jordart'].isin(bedr_cover)]

# Plot the filtered polygons based on the 'jordart' column
gdf_filtered_classes.plot(column='jordart', figsize=(20,12), legend=False)
plt.title('Bedrock & thin cover')


# Display the plot
plt.show()


#%% Import granada data 

# IMPORTANT!! RUN Granada_plotting.py FIRST

import os
import sys

# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
granada_path = os.path.join(os.path.dirname(script_path), '..', 'granada')
sys.path.append(granada_path)

obs_xyz_path = os.path.join(granada_path, 'OBS_XYZ.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ = pd.read_csv(obs_xyz_path)

# Use your_variable in YourScript.py
print(OBS_XYZ)


#%% Filter Polygons based on size

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


### OBS_XYZ needs to be a geodataframe to be counted on GDF
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(OBS_XYZ['Shape_X'], OBS_XYZ['Shape_Y'])]
obs_xyz_gdf = gpd.GeoDataFrame(OBS_XYZ, geometry=geometry, crs=gdf_filtered_classes.crs)



gdf_filtered_classes['polygon_area'] = gdf_filtered_classes.geometry.area

# Set a threshold area. 2 million excludes the biggest
area_threshold = 2000000

# Filter out polygons below the area threshold
gdf_processed = gdf_filtered_classes[gdf_filtered_classes['polygon_area'] <= area_threshold]
gdf_processed.plot(figsize=(20,12))

plt.savefig('gdf_processed.png', dpi=300, bbox_inches='tight')


#%% Plot on gdf file


# Plot the polygons all in one colour
ax = gdf_processed.plot(column='jordart', figsize=(20,12), color='yellow', legend=False)
plt.plot(ax=ax)
# Plot the points on top of the polygons
sctplot = OBS_XYZ.plot.scatter(x='Shape_X', y='Shape_Y', c=OBS_XYZ['blengdber_'], cmap='viridis', ax=ax, s=5, label='OBS_XYZ', legend=False)

plt.savefig('gdf_scatterplot.png', dpi=300, bbox_inches='tight')




#### Find Points inside of a Polygon
ax = gdf_processed.plot(figsize=(18, 11))

# Plotting points on top of polygons
obs_xyz_gdf.plot(ax=ax, color='red', markersize=5)

# Identifying points inside polygons
points_inside = gpd.sjoin(obs_xyz_gdf, gdf_processed, op='within')

# Plotting points inside polygons with a different color
points_inside.plot(ax=ax, color='blue', markersize=5)

plt.legend()

plt.show()




# Plotting polygons
ax = gdf_processed.plot(figsize=(18, 11))

# Identifying polygons with at least one point inside
polygons_with_points = gpd.sjoin(gdf_processed, obs_xyz_gdf, op='intersects')

# Plotting polygons with at least one point inside
polygons_with_points.plot(ax=ax, facecolor='none', edgecolor='yellow')

# Plotting all points on top of polygons
obs_xyz_gdf.plot(ax=ax, color='red', markersize=5)

plt.show()


#%% Areas around Polygon points

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Assuming gdf_processed and obs_xyz_gdf are your GeoDataFrames
# Also assuming you have a given radius (in the same units as your coordinates)

radius = 100  # Adjust this value based on your requirement

# Identifying points inside polygons
points_inside = gpd.sjoin(obs_xyz_gdf, gdf_processed, op='within')

# Create GeoDataFrame for circles around points inside polygons
circles_gdf = gpd.GeoDataFrame(geometry=[Point(xy).buffer(radius) for xy in points_inside['geometry']],
                               crs=obs_xyz_gdf.crs)

# Finding the intersection of circles and polygons
intersection_areas = gpd.overlay(circles_gdf, gdf_processed, how='intersection')

# Plotting polygons
ax = gdf_processed.plot(edgecolor='none', figsize=(20, 12))

# Plotting intersection areas
intersection_areas.plot(ax=ax, facecolor='red', edgecolor='none')

# Plotting points inside polygons with a different color
points_inside.plot(ax=ax, color='blue', markersize=5)

# Plotting circles around points inside polygons
circles_gdf.plot(ax=ax, facecolor='none', edgecolor='none')

plt.show()





#%% Replace point, grid method

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

# Assuming gdf_processed and obs_xyz_gdf are your GeoDataFrames
# Also assuming you have a given radius (in the same units as your coordinates)

# Create a grid over the entire map
# Use min max x y values from bounding box set in the beginning
grid_size = 100  # Adjust this value based on your requirement
x_values = np.arange(minx, maxx, grid_size)
y_values = np.arange(miny, maxy, grid_size)

# Generate a grid of points
grid_points = gpd.GeoDataFrame(geometry=[Point(x, y) for x in x_values for y in y_values],
                               crs=gdf_processed.crs)

# Identify grid points that intersect with the intersection areas
grid_points_inside = gpd.sjoin(grid_points, intersection_areas, op='intersects')

# Remove points inside the intersection areas from obs_xyz_gdf
obs_xyz_gdf = obs_xyz_gdf[~obs_xyz_gdf['geometry'].isin(points_inside['geometry'])]

# Add new points with Z value 0 at the centers of the grid cells
new_points = gpd.GeoDataFrame(geometry=grid_points_inside['geometry'], crs=gdf_processed.crs)
new_points['blengdber_'] = 0


# Extract x, y, and z values from the geometry point
new_points['Shape_X'] = new_points['geometry'].x
new_points['Shape_Y'] = new_points['geometry'].y

# Drop direct copies of points in new_points
new_points = new_points.drop_duplicates(subset=['Shape_X', 'Shape_Y', 'blengdber_'], keep='first')

# Add new points with Z value 0 at the centers of the grid cells to obs_xyz_gdf
obs_xyz_gdf = pd.concat([obs_xyz_gdf, new_points])


# Plotting polygons
ax = gdf_processed.plot(edgecolor='black', figsize=(20, 15))


# Plotting circles around points inside polygons
circles_gdf.plot(ax=ax, facecolor='none', edgecolor='orange')


# Plotting points inside polygons with a different color
points_inside.plot(ax=ax, color='blue', markersize=3)

# Plotting intersection areas in red
intersection_areas.plot(ax=ax, facecolor='red', edgecolor='red')

# Plotting new points at the centers of grid cells
new_points.plot(ax=ax, color='green', markersize=5)

plt.savefig('imputed_boreholes.png', dpi=300, bbox_inches='tight')

plt.show()


#%% Plot the final preprocessed dataset



obs_xyz_gdf.to_csv('OBS_XYZ_gdf.csv', index=False)


# Plot the polygons all in one colour
ax = gdf_processed.plot(column='jordart', figsize=(19,12), color='yellow', legend=False)
plt.plot(ax=ax)
# Plot the points on top of the polygons
sctplot = obs_xyz_gdf.plot.scatter(x='Shape_X', y='Shape_Y', c=obs_xyz_gdf['blengdber_'], cmap='viridis', ax=ax, s=5, label='OBS_XYZ', legend=False)

plt.savefig('preprocessed_boreholes.png', dpi=300, bbox_inches='tight')





# Assuming gdf is your GeoPandas DataFrame
fig, ax = plt.subplots(figsize=(18, 11))

# Create a scatter plot
obs_xyz_gdf.plot.scatter(x='Shape_X', y='Shape_Y', c='blengdber_', cmap='viridis', s=10, alpha=0.7, ax=ax)

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot of Points with Color-Coded Z values')


# Show the plot
plt.show()

