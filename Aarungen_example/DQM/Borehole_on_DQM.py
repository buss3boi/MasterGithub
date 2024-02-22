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


gdf.plot(column='jordart', legend=True)
plt.title('DQM file')
#jestem gupi 
# Now, the variable 'gdf' contains your shapefile data

minx, maxx = 254100, 268000
miny, maxy = 6620100, 6628700

# Filter the GeoDataFrame based on the bounding box
gdf_filtered = gdf.cx[minx:maxx, miny:maxy]
gdf_filtered.plot(column='jordart', legend=False)
plt.title('Filtered DQM file')

# Define the classes you want to display

# Definition 1:
bedr_cover = [12, 13, 43, 100, 101, 110, 130, 140]


# Filter the GeoDataFrame to include only the desired classes
gdf_filtered_classes = gdf_filtered[gdf_filtered['jordart'].isin(bedr_cover)]

# Plot the filtered polygons based on the 'jordart' column
gdf_filtered_classes.plot(column='jordart', legend=False)
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
''

#%% Filter Polygons based on size

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


### OBS_XYZ needs to be a geodataframe to be counted on GDF
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(OBS_XYZ['Shape_X'], OBS_XYZ['Shape_Y'])]
obs_xyz_gdf = gpd.GeoDataFrame(OBS_XYZ, geometry=geometry, crs=gdf_filtered_classes.crs)




gdf_filtered_classes['polygon_area'] = gdf_filtered_classes.geometry.area

# Set a threshold area. 5 million excludes the biggest
area_threshold = 2000000

# Filter out polygons below the area threshold
filtered_gdf = gdf_filtered_classes[gdf_filtered_classes['polygon_area'] <= area_threshold]
filtered_gdf.plot()


### Perform a spatial join
joined_data = gpd.sjoin(filtered_gdf, obs_xyz_gdf, how="inner", op="contains")

# Count the number of points within each polygon
point_count_per_polygon = joined_data.groupby(joined_data.index).size().reset_index(name='point_count')

# Merge the point count back to the GeoDataFrame
gdf_with_point_count = filtered_gdf.merge(point_count_per_polygon, left_index=True, right_on='index')


#%% Plot on gdf file


# Plot the polygons all in one colour
ax = filtered_gdf.plot(column='jordart', color='yellow', legend=False)
plt.plot(ax=ax)
# Plot the points on top of the polygons
sctplot = OBS_XYZ.plot.scatter(x='Shape_X', y='Shape_Y', c=OBS_XYZ['blengdber_'], cmap='viridis', ax=ax, s=3, label='OBS_XYZ', legend=False)

plt.savefig('output_figure.png', dpi=300, bbox_inches='tight')



### Differentiate points on or off the polygons

# Create a new column indicating if a point is within a polygon
obs_xyz_gdf['within_polygon'] = obs_xyz_gdf.index.isin(joined_data.index_right)

# Plot the polygons with color based on point count
ax = filtered_gdf.plot(color='lightgray', legend=True)

# Plot the points on top of the polygons
obs_xyz_gdf.plot.scatter(x='Shape_X', y='Shape_Y', c=obs_xyz_gdf['within_polygon'], cmap='coolwarm', ax=ax, s=3, label='OBS_XYZ', legend=True)
plt.title('Points on a polygon')


### Differentiate polygons with or without points

# Identify polygons that have at least one point from obs_xyz_gdf
polygons_with_points = joined_data['index_right'].unique()

# Create a new column indicating if a polygon has at least one point from obs_xyz_gdf
filtered_gdf['has_points'] = filtered_gdf.index.isin(polygons_with_points)

# Plot the polygons with different colors based on whether they have at least one point
ax = filtered_gdf.plot(column='has_points', cmap='coolwarm')

# Plot the points on top of the polygons
obs_xyz_gdf.plot.scatter(x='Shape_X', y='Shape_Y', c=obs_xyz_gdf['within_polygon'], cmap='coolwarm', ax=ax, s=3, label='OBS_XYZ', legend=True)
plt.title('Polygons with boreholes')



